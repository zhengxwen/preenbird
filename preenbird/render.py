"""Stage 4b — render one segment into a finished vertical Short.

ffmpeg decodes + slows + (HDR->SDR) the segment and streams full frames here; we
crop the head-tracked window per frame (numpy slice) and stream the result to a
second ffmpeg that adds music (or slowed natural audio), an optional caption and
watermark, and hardware-HEVC-encodes the chosen shape (1080x1920 or 1080x1560).

PQ HDR (Canon) is tone-mapped on the GPU with an explicit bt2020->bt709 primaries
conversion + a gentle contrast/saturation bump — without the primaries step the
output looks grey/washed.
"""
from __future__ import annotations

import os
import subprocess

import numpy as np

from .reframe import compute_path

_FONT = "/System/Library/Fonts/Helvetica.ttc"


def _esc(t: str) -> str:
    return str(t).replace("\\", "").replace(":", r"\:").replace("'", "")


def _wm_image(wm) -> str | None:
    if wm and wm.get("enabled"):
        p = wm.get("image")
        if p and os.path.exists(p):
            return p
    return None


def _drawtexts(caption, wm) -> list[str]:
    out = []
    if caption:
        out.append(f"drawtext=fontfile={_FONT}:text='{_esc(caption)}':fontcolor=white:fontsize=52:"
                   f"box=1:boxcolor=black@0.45:boxborderw=16:x=(w-text_w)/2:y=h-200")
    if wm and wm.get("enabled") and wm.get("text") and not _wm_image(wm):
        op = wm.get("opacity", 0.85)
        pos = {"tl": "x=40:y=40", "tr": "x=w-text_w-40:y=40",
               "bl": "x=40:y=h-text_h-40", "br": "x=w-text_w-40:y=h-text_h-40"}.get(
            wm.get("position", "br"), "x=w-text_w-40:y=h-text_h-40")
        out.append(f"drawtext=fontfile={_FONT}:text='{_esc(wm['text'])}':fontcolor=white@{op}:fontsize=40:"
                   f"shadowcolor=black@0.5:shadowx=2:shadowy=2:{pos}")
    return out


def adjust_filters(adjust) -> list[str]:
    """ffmpeg filters for the UI sliders (brightness/contrast/saturation/warmth/sharpen)."""
    if not adjust:
        return []
    b = float(adjust.get("brightness", 0))
    c = float(adjust.get("contrast", 1))
    s = float(adjust.get("saturation", 1))
    g = float(adjust.get("gamma", 1))
    w = float(adjust.get("warmth", 0))
    sh = float(adjust.get("sharpen", 0))
    out = []
    if b or c != 1 or s != 1 or g != 1:
        out.append(f"eq=brightness={b:.3f}:contrast={c:.3f}:saturation={s:.3f}:gamma={g:.3f}")
    if w:
        out.append(f"colorbalance=rm={w:.3f}:gm=0:bm={-w:.3f}")
    if sh:
        out.append(f"unsharp=5:5:{sh:.3f}:5:5:0.0")
    return out


_CRF = {"x265": {"high": 20, "medium": 24, "low": 28},
        "x264": {"high": 19, "medium": 23, "low": 27},
        "av1": {"high": 28, "medium": 34, "low": 40}}
_VT_BR = {"high": "12M", "medium": "8M", "low": "5M"}
_COLOR_TAGS = ["-pix_fmt", "yuv420p", "-color_range", "tv", "-colorspace", "bt709",
               "-color_primaries", "bt709", "-color_trc", "bt709"]


def video_codec_args(codec, quality, render_cfg):
    """Encoder flags: hevc_vt (fast HW HEVC), x265/x264 (software CRF), av1."""
    codec = codec or render_cfg.get("codec", "hevc_vt")
    quality = quality or render_cfg.get("quality", "high")
    if codec == "x265":
        return ["-c:v", "libx265", "-crf", str(_CRF["x265"].get(quality, 22)),
                "-preset", "medium", "-tag:v", "hvc1"] + _COLOR_TAGS
    if codec == "x264":
        return ["-c:v", "libx264", "-crf", str(_CRF["x264"].get(quality, 21)),
                "-preset", "medium"] + _COLOR_TAGS
    if codec == "av1":
        return ["-c:v", "libsvtav1", "-crf", str(_CRF["av1"].get(quality, 34)),
                "-preset", "6"] + _COLOR_TAGS
    return ["-c:v", "hevc_videotoolbox", "-b:v",
            _VT_BR.get(quality, str(render_cfg.get("bitrate", "8M"))),
            "-tag:v", "hvc1"] + _COLOR_TAGS


def render_segment(info, seg, render_cfg: dict, out_path: str, *,
                   music: str | None = None, mute: bool = False, zoom: float = 1.0,
                   caption: str | None = None, watermark: dict | None = None,
                   adjust: dict | None = None, codec: str | None = None,
                   quality: str | None = None, music_vol: float | None = None,
                   natural_vol: float | None = None, progress=None, should_cancel=None) -> str:
    W, H = info.width, info.height
    out_w, out_h = render_cfg["width"], render_cfg["height"]
    speed = render_cfg["speed"]
    cfgfps = render_cfg.get("fps")
    out_fps = max(1, round(info.fps * speed)) if cfgfps in (None, "auto", "") else int(cfgfps)
    fade = render_cfg.get("fade_s", 0.4)

    rc = dict(render_cfg)
    rc["fps"] = out_fps
    path = compute_path(seg, W, H, rc, zoom=zoom)
    cw, ch = path["crop_w"], path["crop_h"]
    s = out_h / ch
    Ws = max(out_w, int(round(W * s)))
    Hs = max(out_h, int(round(H * s)))
    out_dur = path["n"] / out_fps

    # --- decode: slow-mo + fps + prescale (+ HDR->SDR) -> raw BGR frames ---
    use_ocl = getattr(info, "transfer", "") == "smpte2084"   # PQ HDR
    vf = [f"setpts=PTS/{speed}", f"fps={out_fps}"]
    if use_ocl:
        tm = render_cfg.get("tonemap", "hable")
        con = render_cfg.get("hdr_contrast", 1.06)
        sat = render_cfg.get("hdr_saturation", 1.15)
        vf += ["format=p010", "hwupload",
               f"tonemap_opencl=tonemap={tm}:desat=0:peak=100:"
               f"primaries=bt709:transfer=bt709:matrix=bt709:format=nv12",
               "hwdownload", "format=nv12", f"eq=contrast={con}:saturation={sat}",
               f"scale={Ws}:{Hs}", "format=bgr24"]
    elif info.hdr:                                            # HLG and other HDR
        vf += [f"scale={Ws}:{Hs}", "colorspace=all=bt709:iall=bt2020:fast=1", "format=bgr24"]
    else:
        vf += [f"scale={Ws}:{Hs}", "format=bgr24"]
    adj = adjust_filters(adjust)
    if adj:                                              # user grade: brightness/color/sharpen
        vf = vf[:-1] + adj + vf[-1:]                     # insert before the final format=bgr24
    dec = ["ffmpeg", "-v", "error"]
    if use_ocl:
        dec += ["-init_hw_device", "opencl=ocl", "-filter_hw_device", "ocl"]
    dec += ["-ss", f"{seg.start:.3f}", "-t", f"{seg.end - seg.start:.3f}",
            "-i", info.path, "-vf", ",".join(vf),
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]

    # --- encoder inputs: pipe video (0) [+ natural audio] [+ music] [+ watermark] ---
    enc = ["ffmpeg", "-v", "error", "-y",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{out_w}x{out_h}",
           "-r", str(out_fps), "-i", "pipe:0"]
    mvol = render_cfg.get("music_vol", 0.85) if music_vol is None else float(music_vol)
    nvol = render_cfg.get("natural_vol", 0.2) if natural_vol is None else float(natural_vol)
    want_nat = (not mute) and nvol > 0 and info.has_audio
    want_mus = (not mute) and bool(music) and mvol > 0
    idx, na_idx, mu_idx, w_idx = 1, None, None, None
    if want_nat:
        enc += ["-ss", f"{seg.start:.3f}", "-t", f"{seg.end - seg.start:.3f}", "-i", info.path]
        na_idx = idx
        idx += 1
    if want_mus:
        enc += ["-stream_loop", "-1", "-i", music]
        mu_idx = idx
        idx += 1
    wm_img = _wm_image(watermark)
    if wm_img:
        enc += ["-i", wm_img]
        w_idx = idx
        idx += 1

    chains, vlabel, alabel = [], None, None
    draws = _drawtexts(caption, watermark)
    if wm_img:
        op = watermark.get("opacity", 0.85)
        sc = watermark.get("scale", 0.14)
        m = int(round(out_w * 0.04))
        xy = {"tl": f"{m}:{m}", "tr": f"W-w-{m}:{m}", "bl": f"{m}:H-h-{m}",
              "br": f"W-w-{m}:H-h-{m}"}.get(watermark.get("position", "br"), f"W-w-{m}:H-h-{m}")
        chains.append("[0:v]" + (",".join(draws) if draws else "null") + "[vb]")
        chains.append(f"[{w_idx}:v]format=rgba,colorchannelmixer=aa={op},scale={int(out_w * sc)}:-1[wm]")
        chains.append(f"[vb][wm]overlay={xy}[v]")
        vlabel = "[v]"
    elif draws:
        chains.append("[0:v]" + ",".join(draws) + "[v]")
        vlabel = "[v]"

    fade_out_st = f"{max(0.0, out_dur - fade):.3f}"
    if want_nat:
        chains.append(f"[{na_idx}:a]atempo={speed},volume={nvol:.3f}[na]")
    if want_mus:
        chains.append(f"[{mu_idx}:a]loudnorm=I={render_cfg.get('music_lufs', -16)}:TP=-1.5,volume={mvol:.3f}[mu]")
    afinal = ("amx" if (want_nat and want_mus) else "na" if want_nat else "mu" if want_mus else None)
    if want_nat and want_mus:
        chains.append("[na][mu]amix=inputs=2:duration=first:normalize=0[amx]")
    if afinal:
        chains.append(f"[{afinal}]afade=t=in:st=0:d={fade},afade=t=out:st={fade_out_st}:d={fade},aresample=48000[aout]")
        alabel = "[aout]"

    if chains:
        enc += ["-filter_complex", ";".join(chains)]
    enc += (["-map", vlabel] if vlabel else ["-map", "0:v:0"])
    enc += (["-map", alabel, "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-shortest"]
            if alabel else ["-an"])
    enc += video_codec_args(codec, quality, render_cfg) + ["-t", f"{out_dur:.3f}", out_path]

    # --- run the two-process pipeline ---
    dec_p = subprocess.Popen(dec, stdout=subprocess.PIPE, bufsize=10 ** 8)
    enc_p = subprocess.Popen(enc, stdin=subprocess.PIPE, bufsize=10 ** 8)
    frame_bytes = Ws * Hs * 3
    xs, ys, n = path["xs"], path["ys"], path["n"]
    i = 0
    try:
        while True:
            buf = dec_p.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                break
            frame = np.frombuffer(buf, dtype=np.uint8).reshape(Hs, Ws, 3)
            j = min(i, n - 1)
            x = min(max(0, int(round(xs[j] * s))), Ws - out_w)
            y = min(max(0, int(round(ys[j] * s))), Hs - out_h)
            enc_p.stdin.write(frame[y:y + out_h, x:x + out_w].tobytes())
            i += 1
            if i % 15 == 0:
                if progress:
                    progress(i, n)
                if should_cancel and should_cancel():
                    dec_p.terminate()
                    enc_p.terminate()
                    raise RuntimeError("cancelled")
    finally:
        if progress:
            progress(min(i, n), n)
        dec_p.stdout.close()
        try:
            enc_p.stdin.close()
        except BrokenPipeError:
            pass
        dec_p.wait()
        enc_p.wait()
    return out_path
