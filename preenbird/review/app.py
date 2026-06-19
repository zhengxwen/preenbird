"""Stage 3 — local web UI to pick & trim which segments to publish.

Lists every scored candidate (ranked), plays a browser-friendly preview proxy
transcoded on demand from the HEVC/HDR source, lets you check what to publish and
set in/out points, then renders the selection via stage 4.
"""
from __future__ import annotations

import os
import subprocess
import threading
import uuid

from fastapi import Body, FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from ..probe import VideoInfo
from ..store import load_candidates

JOBS = {}                                                # render job_id -> live progress dict


def _work_dir(cfg):
    return cfg["paths"]["work_dir"]


def _stems(cfg):
    wd = _work_dir(cfg)
    if not os.path.isdir(wd):
        return []
    return [s for s in sorted(os.listdir(wd))
            if os.path.exists(os.path.join(wd, s, "candidates.json"))]


def _preview_path(cfg, stem: str, idx: int) -> str:
    return os.path.join(_work_dir(cfg), stem, "preview", f"seg_{idx:03d}.mp4")


def _highlight(seg, maxlen=14.0):
    """Auto highlight window (start, end): ~maxlen s centred on the sharpest-detected
    frame, clamped to the segment. Used for the preview proxy AND as the default in/out
    so the preview matches what will be rendered."""
    dur = seg.end - seg.start
    if dur <= maxlen or not seg.track:
        return seg.start, seg.end
    best = max(seg.track, key=lambda d: d["conf"])["t"]
    start = min(max(seg.start, best - maxlen / 2), max(seg.start, seg.end - maxlen))
    return start, start + min(maxlen, seg.end - start)


def _ensure_preview(cfg, stem: str, info: VideoInfo, seg) -> str:
    out = _preview_path(cfg, stem, idx=seg.idx)
    cand = os.path.join(_work_dir(cfg), stem, "candidates.json")
    if os.path.exists(out) and os.path.getmtime(out) >= os.path.getmtime(cand):
        return out
    os.makedirs(os.path.dirname(out), exist_ok=True)

    # preview only the auto-selected highlight window (a full 86s 4K-HDR clip would
    # take ~100s to transcode); this is the same window used as the default in/out.
    pstart, pend = _highlight(seg, cfg["score"].get("highlight_s", 14.0))
    pdur = pend - pstart

    use_ocl = getattr(info, "transfer", "") == "smpte2084"
    if use_ocl:
        vf = ["fps=20", "format=p010", "hwupload",
              "tonemap_opencl=tonemap=hable:desat=0:peak=100:"
              "primaries=bt709:transfer=bt709:matrix=bt709:format=nv12",
              "hwdownload", "format=nv12", "eq=contrast=1.06:saturation=1.15",
              "scale=400:-2", "format=yuv420p"]
    elif info.hdr:
        vf = ["fps=20", "scale=400:-2", "colorspace=all=bt709:iall=bt2020:fast=1", "format=yuv420p"]
    else:
        vf = ["fps=20", "scale=400:-2", "format=yuv420p"]
    cmd = ["ffmpeg", "-v", "error", "-y"]
    if use_ocl:
        cmd += ["-init_hw_device", "opencl=ocl", "-filter_hw_device", "ocl"]
    cmd += ["-ss", f"{pstart:.3f}", "-t", f"{pdur:.3f}", "-i", info.path,
            "-vf", ",".join(vf), "-c:v", "h264_videotoolbox", "-b:v", "1400k",
            "-pix_fmt", "yuv420p", "-color_range", "tv", "-an", "-movflags", "+faststart", out]
    subprocess.run(cmd, check=True, capture_output=True)
    return out


def create_app(cfg) -> FastAPI:
    app = FastAPI(title="Preen Review")
    static_dir = os.path.join(os.path.dirname(__file__), "static")
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/")
    def index():
        return FileResponse(os.path.join(static_dir, "index.html"))

    @app.get("/api/candidates")
    def candidates():
        groups = []
        for stem in _stems(cfg):
            info, segs = load_candidates(os.path.join(_work_dir(cfg), stem))
            items = []
            for s in segs:
                d = s.to_dict()
                hi_s, hi_e = _highlight(s, cfg["score"].get("highlight_s", 14.0))
                d["clip_in"], d["clip_out"] = round(hi_s, 1), round(hi_e, 1)
                d["thumb"] = f"/api/thumb/{stem}/{s.idx}"
                d["preview"] = f"/api/preview/{stem}/{s.idx}"
                items.append(d)
            items.sort(key=lambda x: (x.get("publish_score") or 0), reverse=True)
            groups.append({"stem": stem, "video": os.path.basename(info.path),
                           "fps": info.fps, "hdr": info.hdr, "segments": items})
        return {"groups": groups}

    @app.get("/api/thumb/{stem}/{idx}")
    def thumb(stem: str, idx: int):
        p = os.path.join(_work_dir(cfg), stem, "thumbs", f"seg_{idx:03d}.jpg")
        return FileResponse(p) if os.path.exists(p) else JSONResponse({"error": "no thumb"}, 404)

    @app.get("/api/preview/{stem}/{idx}")
    def preview(stem: str, idx: int):
        info, segs = load_candidates(os.path.join(_work_dir(cfg), stem))
        seg = next((s for s in segs if s.idx == idx), None)
        if seg is None:
            return JSONResponse({"error": "no segment"}, 404)
        try:
            return FileResponse(_ensure_preview(cfg, stem, info, seg), media_type="video/mp4")
        except subprocess.CalledProcessError as e:
            return JSONResponse({"error": e.stderr.decode()[-400:]}, 500)

    @app.get("/api/output/{stem}/{idx}")
    def output(stem: str, idx: int):
        p = os.path.join(_work_dir(cfg), stem, "out", f"short_{idx:03d}.mp4")
        return FileResponse(p) if os.path.exists(p) else JSONResponse({"error": "not rendered"}, 404)

    @app.get("/api/source/{stem}")
    def source_proxy(stem: str):
        from ..media import hdr_to_sdr
        wd = os.path.join(_work_dir(cfg), stem)
        cand = os.path.join(wd, "candidates.json")
        if not os.path.exists(cand):
            return JSONResponse({"error": "unknown"}, 404)
        out = os.path.join(wd, "source.mp4")          # whole-clip low-res proxy for scrubbing
        if not (os.path.exists(out) and os.path.getmtime(out) >= os.path.getmtime(cand)):
            info, _ = load_candidates(wd)
            g, pre, post = hdr_to_sdr(info)
            vf = ["fps=24"] + list(pre) + ["scale=480:-2"] + list(post) + ["format=yuv420p"]
            cmd = (["ffmpeg", "-v", "error", "-y"] + g + ["-i", info.path, "-vf", ",".join(vf),
                   "-c:v", "h264_videotoolbox", "-b:v", "2000k", "-pix_fmt", "yuv420p",
                   "-color_range", "tv", "-an", "-movflags", "+faststart", out])
            subprocess.run(cmd, check=True, capture_output=True)
        return FileResponse(out, media_type="video/mp4")

    @app.post("/api/render")
    def render(payload: dict = Body(...)):
        from ..render import render_segment
        items = payload.get("items", [])
        if not items:
            return {"error": "no items"}
        zoom = float(payload.get("zoom", 1.0))
        mute = bool(payload.get("mute", False))
        want_caption = bool(payload.get("caption", True))
        rcfg = dict(cfg["render"])
        if payload.get("aspect") == "1080x1560":
            rcfg["width"], rcfg["height"] = 1080, 1560
        elif payload.get("aspect") == "9:16":
            rcfg["width"], rcfg["height"] = 1080, 1920
        wm = dict(cfg["render"].get("watermark") or {})
        if payload.get("watermark"):
            wm["enabled"] = True
        codec, quality = payload.get("codec"), payload.get("quality")
        music_vol, natural_vol = payload.get("music_vol"), payload.get("natural_vol")
        mdir = cfg["paths"]["music_dir"]

        def resolve_music(rel):
            if not rel:
                return None
            p = os.path.realpath(os.path.join(mdir, rel))
            return p if os.path.exists(p) else None

        jid = uuid.uuid4().hex[:8]
        job = {"total": len(items), "done": 0, "cur": "", "frac": 0.0,
               "results": [], "finished": False, "cancel": False}
        JOBS[jid] = job

        def run():
            for it in items:
                if job["cancel"]:
                    break
                stem = it["stem"]
                wd = os.path.join(_work_dir(cfg), stem)
                info, segs = load_candidates(wd)
                seg = next((s for s in segs if s.idx == int(it["seg"])), None)
                if seg is None:
                    job["done"] += 1
                    continue
                seg.start = float(it.get("start", seg.start))
                seg.end = float(it.get("end", seg.end))
                caption = seg.species if (want_caption and seg.species) else None
                music = None if mute else resolve_music(it.get("music"))
                job["cur"] = f"{seg.species or 'bird'} #{seg.idx}"
                job["frac"] = 0.0
                os.makedirs(os.path.join(wd, "out"), exist_ok=True)
                out = os.path.join(wd, "out", f"short_{seg.idx:03d}.mp4")
                try:
                    render_segment(info, seg, rcfg, out, music=music, mute=mute, zoom=zoom,
                                   caption=caption, watermark=wm, adjust=it.get("adjust"),
                                   codec=codec, quality=quality, music_vol=music_vol,
                                   natural_vol=natural_vol,
                                   progress=lambda fr, tot: job.update(frac=fr / max(1, tot)),
                                   should_cancel=lambda: job["cancel"])
                    job["results"].append({"stem": stem, "seg": seg.idx, "ok": True,
                                           "url": f"/api/output/{stem}/{seg.idx}"})
                except Exception as e:  # noqa: BLE001
                    cancelled = job["cancel"] or "cancel" in str(e).lower()
                    job["results"].append({"stem": stem, "seg": seg.idx, "ok": False,
                                           "error": "cancelled" if cancelled else str(e)})
                job["done"] += 1
            job["finished"] = True

        threading.Thread(target=run, daemon=True).start()
        return {"job": jid}

    @app.get("/api/render_status/{jid}")
    def render_status(jid: str):
        return JOBS.get(jid, {"error": "unknown job", "finished": True})

    @app.get("/api/music")
    def music_list():
        from ..music import list_tracks
        base = cfg["paths"]["music_dir"]
        out = []
        for t in list_tracks(base):
            rel = os.path.relpath(t, base)
            out.append({"path": rel, "name": os.path.splitext(os.path.basename(t))[0],
                        "dir": os.path.dirname(rel) or "."})
        return {"tracks": out}

    @app.get("/api/music_file/{relpath:path}")
    def music_file(relpath: str):
        base = os.path.realpath(cfg["paths"]["music_dir"])
        p = os.path.realpath(os.path.join(base, relpath))
        if not p.startswith(base + os.sep) or not os.path.exists(p):
            return JSONResponse({"error": "not found"}, 404)
        return FileResponse(p)

    @app.post("/api/render_cancel/{jid}")
    def render_cancel(jid: str):
        job = JOBS.get(jid)
        if job:
            job["cancel"] = True
        return {"ok": bool(job)}

    @app.post("/api/export")
    def export(payload: dict = Body(...)):
        from ..fcpxml import generate_fcpxml
        zoom = float(payload.get("zoom", 1.0))
        rcfg = dict(cfg["render"])
        if payload.get("aspect") == "1080x1560":
            rcfg["width"], rcfg["height"] = 1080, 1560
        elif payload.get("aspect") == "9:16":
            rcfg["width"], rcfg["height"] = 1080, 1920
        by_stem = {}
        for it in payload.get("items", []):
            by_stem.setdefault(it["stem"], []).append(it)
        results = []
        for stem, its in by_stem.items():
            wd = os.path.join(_work_dir(cfg), stem)
            info, segs = load_candidates(wd)
            segmap = {s.idx: s for s in segs}
            fitems = []
            for it in its:
                seg = segmap.get(int(it["seg"]))
                if seg is None:
                    continue
                fitems.append({"seg": seg, "start": float(it.get("start", seg.start)),
                               "end": float(it.get("end", seg.end)),
                               "name": f"{seg.species or 'bird'}_{seg.idx:02d}"})
            os.makedirs(os.path.join(wd, "out"), exist_ok=True)
            out = os.path.join(wd, "out", f"{stem}.fcpxml")
            generate_fcpxml(info, fitems, out, rcfg, zoom=zoom)
            results.append({"stem": stem, "path": out, "n": len(fitems)})
        return {"results": results}

    return app
