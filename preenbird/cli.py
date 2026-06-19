"""preen command-line interface.

  preen detect <video...>     # stage 1: cut out the no-bird parts
  preen score  <video...>     # stage 2: identify + rate each segment
  preen scan   <video...>     # detect + score in one go
  preen review                # stage 3: pick what to publish (web UI) [later]
  preen render                # stage 4: tracking 9:16 + 0.5x + music  [later]
"""
from __future__ import annotations

import argparse
import os
import sys

from . import __version__
from .config import load_config


def _fmt_dur(s: float) -> str:
    m, sec = divmod(int(round(s)), 60)
    return f"{m:d}:{sec:02d}"


def cmd_detect(args, cfg):
    from .detect import detect_video, load_detector
    from .media import extract_frame
    from .probe import probe
    from .store import save_candidates, work_dir_for

    print(f"Loading detector {cfg['models']['detector']} on {cfg['detect']['device']} ...")
    model, bird_class = load_detector(cfg["models"]["detector"], cfg["detect"]["device"])

    for video in args.videos:
        if not os.path.exists(video):
            print(f"  !! not found: {video}", file=sys.stderr)
            continue
        info = probe(video)
        print(f"\n=== {os.path.basename(video)} "
              f"[{info.width}x{info.height} {info.fps}fps {_fmt_dur(info.duration)}] ===")
        segments = detect_video(
            info, cfg["detect"], model, bird_class,
            progress=lambda p, t: print(f"\r  scanning {p}/{t} frames ({100 * p // t}%)",
                                        end="", flush=True))
        print()
        work_dir = work_dir_for(cfg, video)

        kept = 0.0
        for seg in segments:
            kept += seg.duration
            thumb = os.path.join(work_dir, "thumbs", f"seg_{seg.idx:03d}.jpg")
            best = max(seg.track, key=lambda d: d["conf"]) if seg.track else None
            extract_frame(info, best["t"] if best else seg.start, thumb,
                          scale_width=640, hwaccel=cfg["detect"]["hwaccel"])
            seg.thumb = os.path.relpath(thumb, work_dir)

        save_candidates(work_dir, info, segments)
        ratio = (kept / info.duration * 100) if info.duration else 0
        print(f"  {len(segments)} bird segments, {_fmt_dur(kept)} kept "
              f"({ratio:.0f}% of source) -> {work_dir}/candidates.json")
        for seg in segments:
            print(f"    #{seg.idx:02d}  {_fmt_dur(seg.start)}–{_fmt_dur(seg.end)}  "
                  f"{seg.duration:4.1f}s  conf={seg.peak_conf:.2f}  size={seg.max_area_frac*100:.1f}%")


def cmd_score(args, cfg):
    from .ollama_vlm import health
    from .score import score_segment
    from .store import load_candidates, save_candidates, work_dir_for

    ok, msg = health()
    if not ok:
        print(f"!! Ollama not reachable: {msg}", file=sys.stderr)
        return 1
    model = args.model or (cfg["models"]["vlm_high"] if args.high else cfg["models"]["vlm"])
    nf = args.frames or cfg["score"].get("frames_per_segment", 5)
    print(f"Scoring with {model}  ({nf} frame(s)/segment)")

    for video in args.videos:
        work_dir = work_dir_for(cfg, video)
        cand = os.path.join(work_dir, "candidates.json")
        if not os.path.exists(cand):
            print(f"  !! no candidates for {os.path.basename(video)} (run detect first)",
                  file=sys.stderr)
            continue
        info, segments = load_candidates(work_dir)
        print(f"\n=== {os.path.basename(video)}: scoring {len(segments)} segments ===")
        N = len(segments)
        for i, seg in enumerate(segments, 1):
            score_segment(info, seg, cfg["score"], model, work_dir,
                          hwaccel=cfg["detect"]["hwaccel"], n_frames=args.frames,
                          progress=lambda j, f, i=i: print(
                              f"\r  [{model}] scoring {i}/{N} · frame {j}/{f} ...",
                              end="", flush=True))
            if seg.vlm_error:
                print(f"\r    #{seg.idx:02d}  VLM error: {seg.vlm_error}          ")
            else:
                print(f"\r    #{seg.idx:02d}  {(seg.species or '?'):<22} "
                      f"score={seg.publish_score:5.1f}  "
                      f"beauty={seg.beauty} interest={seg.interestingness}  "
                      f"{'[common]' if seg.common_lbj else ''} {seg.behavior or ''}")
        save_candidates(work_dir, info, segments)

        ranked = sorted([s for s in segments if s.publish_score is not None],
                        key=lambda s: s.publish_score, reverse=True)
        print("  --- ranked ---")
        for s in ranked:
            print(f"    {s.publish_score:5.1f}  #{s.idx:02d}  {s.species or '?'}"
                  f"  ({_fmt_dur(s.start)}, {s.duration:.1f}s)")


def cmd_scan(args, cfg):
    cmd_detect(args, cfg)
    return cmd_score(args, cfg)


def cmd_review(args, cfg):
    import threading
    import webbrowser

    import uvicorn

    from .review.app import create_app
    app = create_app(cfg)
    url = f"http://127.0.0.1:{args.port}"
    print(f"Preen review UI → {url}   (Ctrl-C to stop)")
    if not args.no_open:                                  # open the browser once the server is up
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


def _select_segments(segments, args):
    if args.seg:
        want = set(args.seg)
        return [s for s in segments if s.idx in want]
    scored = [s for s in segments if s.publish_score is not None]
    pool = scored or segments
    pool = sorted(pool, key=lambda s: (s.publish_score or 0), reverse=True)
    if args.all:
        return pool
    return pool[:args.top]


def cmd_render(args, cfg):
    from .music import list_tracks, pick_track
    from .render import render_segment
    from .store import load_candidates, work_dir_for

    rcfg = dict(cfg["render"])
    if args.aspect == "9:16":
        rcfg["width"], rcfg["height"] = 1080, 1920
    elif args.aspect == "1080x1560":
        rcfg["width"], rcfg["height"] = 1080, 1560
    if args.fast:
        rcfg["tracking"] = False
    wm = dict(cfg["render"].get("watermark") or {})
    if args.watermark:
        wm["enabled"] = True
    adjust = {"brightness": args.brightness, "contrast": args.contrast,
              "saturation": args.saturation, "sharpen": args.sharpen, "warmth": args.warmth}
    if args.mute or args.no_music:
        music = None
    elif args.music:
        music = next((t for t in list_tracks(cfg["paths"]["music_dir"])
                      if args.music in (os.path.basename(t), t)), None)
        if music is None and os.path.exists(args.music):
            music = args.music
        if music is None:
            print(f"  (music '{args.music}' not found; rendering without it)", file=sys.stderr)
    else:
        music = pick_track(cfg["paths"]["music_dir"])

    for video in args.videos:
        work_dir = work_dir_for(cfg, video)
        if not os.path.exists(os.path.join(work_dir, "candidates.json")):
            print(f"  !! no candidates for {os.path.basename(video)} (run scan first)",
                  file=sys.stderr)
            continue
        info, segments = load_candidates(work_dir)
        sel = _select_segments(segments, args)
        if not sel:
            print(f"  !! nothing selected for {os.path.basename(video)}", file=sys.stderr)
            continue
        if args.start is not None:                       # manual trim override
            for s in sel:
                s.start = max(0.0, args.start)
                s.end = min(info.duration, s.start + (args.dur or (s.end - s.start)))
        outdir = os.path.join(work_dir, "out")
        os.makedirs(outdir, exist_ok=True)
        print(f"\n=== {os.path.basename(video)}: rendering {len(sel)} short(s) ===")
        for seg in sel:
            caption = None
            if cfg["render"].get("caption_species") and not args.no_caption and seg.species:
                caption = seg.species
            out = os.path.join(outdir, f"short_{seg.idx:03d}.mp4")
            print(f"  #{seg.idx:02d} {seg.species or 'bird'} "
                  f"({seg.duration:.1f}s -> {seg.duration / rcfg['speed']:.1f}s @0.5x) ...")
            render_segment(info, seg, rcfg, out, music=music, mute=args.mute,
                           zoom=args.zoom, caption=caption, watermark=wm, adjust=adjust,
                           codec=args.codec, quality=args.quality,
                           music_vol=args.music_vol, natural_vol=args.natural_vol,
                           progress=lambda fr, tot: print(f"\r     {fr}/{tot} frames", end="", flush=True))
            print(f"\r     -> {out}              ")


def cmd_export(args, cfg):
    from .fcpxml import generate_fcpxml
    from .store import load_candidates, work_dir_for

    rcfg = dict(cfg["render"])
    if args.aspect == "9:16":
        rcfg["width"], rcfg["height"] = 1080, 1920
    elif args.aspect == "1080x1560":
        rcfg["width"], rcfg["height"] = 1080, 1560

    for video in args.videos:
        work_dir = work_dir_for(cfg, video)
        if not os.path.exists(os.path.join(work_dir, "candidates.json")):
            print(f"  !! no candidates for {os.path.basename(video)} (run scan first)", file=sys.stderr)
            continue
        info, segments = load_candidates(work_dir)
        sel = _select_segments(segments, args)
        if not sel:
            print(f"  !! nothing selected for {os.path.basename(video)}", file=sys.stderr)
            continue
        items = []
        for seg in sel:
            if args.start is not None:
                st = max(0.0, args.start)
                en = min(info.duration, st + (args.dur or (seg.end - seg.start)))
            else:
                st, en = seg.start, seg.end
            items.append({"seg": seg, "start": st, "end": en,
                          "name": f"{seg.species or 'bird'}_{seg.idx:02d}"})
        outdir = os.path.join(work_dir, "out")
        os.makedirs(outdir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(video))[0]
        out = os.path.join(outdir, f"{stem}.fcpxml")
        generate_fcpxml(info, items, out, rcfg, zoom=args.zoom)
        print(f"\n=== {os.path.basename(video)}: {len(items)} project(s) -> {out} ===")
        print(f"    open -a 'Final Cut Pro' '{out}'   (then select the clip and set 50% retime for slow-mo)")


def main(argv=None):
    p = argparse.ArgumentParser(prog="preen", description="Auto-edit bird footage into YouTube Shorts.")
    p.add_argument("--config", help="path to config.yaml (defaults built in)")
    p.add_argument("--version", action="version", version=f"preen {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    for name in ("detect", "scan"):
        sp = sub.add_parser(name)
        sp.add_argument("videos", nargs="+")
        sp.add_argument("--high", action="store_true", help="use the higher-accuracy VLM (scan/score)")
        if name == "scan":
            sp.add_argument("--model", help="Ollama vision model name (overrides config)")
            sp.add_argument("--frames", type=int, help="frames per segment fed to the VLM")

    sp = sub.add_parser("score")
    sp.add_argument("videos", nargs="+")
    sp.add_argument("--high", action="store_true", help="use the higher-accuracy VLM")
    sp.add_argument("--model", help="Ollama vision model name (overrides config)")
    sp.add_argument("--frames", type=int, help="frames per segment fed to the VLM")

    rvp = sub.add_parser("review")
    rvp.add_argument("--port", type=int, default=8765)
    rvp.add_argument("--no-open", action="store_true", help="don't auto-open the browser")

    rp = sub.add_parser("render")
    rp.add_argument("videos", nargs="+")
    rp.add_argument("--seg", type=int, nargs="*", help="segment indices to render")
    rp.add_argument("--top", type=int, default=1, help="render the N highest-scored segments")
    rp.add_argument("--all", action="store_true", help="render every segment")
    rp.add_argument("--zoom", type=float, default=1.0, help="<1 punches in (tracking on portrait)")
    rp.add_argument("--mute", action="store_true", help="no audio")
    rp.add_argument("--no-caption", action="store_true", help="don't overlay species name")
    rp.add_argument("--fast", action="store_true", help="disable bird tracking (static center crop)")
    rp.add_argument("--start", type=float, help="override: clip start time (s) within the source")
    rp.add_argument("--dur", type=float, help="override: clip duration (s) from --start")
    rp.add_argument("--aspect", choices=["9:16", "1080x1560"], help="output shape (default 1080x1920)")
    rp.add_argument("--watermark", action="store_true", help="overlay the configured watermark")
    rp.add_argument("--brightness", type=float, default=0.0, help="-0.3..0.3")
    rp.add_argument("--contrast", type=float, default=1.0, help="0.5..2.0")
    rp.add_argument("--saturation", type=float, default=1.0, help="0..2.0")
    rp.add_argument("--sharpen", type=float, default=0.0, help="0..3 (unsharp amount)")
    rp.add_argument("--warmth", type=float, default=0.0, help="-0.3..0.3 (warm/cool)")
    rp.add_argument("--codec", choices=["hevc_vt", "x265", "x264", "av1"], help="video codec")
    rp.add_argument("--quality", choices=["high", "medium", "low"], help="encode quality")
    rp.add_argument("--music", help="backing track filename (in library) or path")
    rp.add_argument("--no-music", action="store_true", help="no backing music")
    rp.add_argument("--music-vol", type=float, help="0..1 music level")
    rp.add_argument("--natural-vol", type=float, help="0..1 original-audio level")

    ep = sub.add_parser("export")
    ep.add_argument("videos", nargs="+")
    ep.add_argument("--seg", type=int, nargs="*", help="segment indices to export")
    ep.add_argument("--top", type=int, default=1, help="export the N highest-scored segments")
    ep.add_argument("--all", action="store_true", help="export every segment")
    ep.add_argument("--zoom", type=float, default=1.0)
    ep.add_argument("--start", type=float, help="override clip start (s)")
    ep.add_argument("--dur", type=float, help="override clip duration (s)")
    ep.add_argument("--aspect", choices=["9:16", "1080x1560"])

    args = p.parse_args(argv)
    cfg = load_config(args.config or ("config.yaml" if os.path.exists("config.yaml") else None))

    return {
        "detect": cmd_detect, "score": cmd_score, "scan": cmd_scan,
        "review": cmd_review, "render": cmd_render, "export": cmd_export,
    }[args.cmd](args, cfg) or 0


if __name__ == "__main__":
    sys.exit(main())
