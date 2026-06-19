"""ffmpeg helpers for pulling stills and crops out of source video."""
from __future__ import annotations

import os
import subprocess


def hdr_to_sdr(info):
    """Return (global_args, filters_before_scale, filters_after_scale) for HDR->SDR.

    PQ (smpte2084) needs GPU tone-mapping WITH a bt2020->bt709 gamut conversion
    (without it the image is grey/washed); HLG degrades fine through `colorspace`.
    Used by stills, crops, the preview proxy and the final render so colour is
    consistent everywhere.
    """
    transfer = getattr(info, "transfer", "")
    if transfer == "smpte2084":
        return (["-init_hw_device", "opencl=ocl", "-filter_hw_device", "ocl"],
                ["format=p010", "hwupload",
                 "tonemap_opencl=tonemap=hable:desat=0:peak=100:"
                 "primaries=bt709:transfer=bt709:matrix=bt709:format=nv12",
                 "hwdownload", "format=nv12", "eq=contrast=1.06:saturation=1.15"],
                [])
    if getattr(info, "hdr", False):
        return ([], [], ["colorspace=all=bt709:iall=bt2020:fast=1"])
    return ([], [], [])


def extract_frame(info, t: float, out_path: str,
                  scale_width: int | None = None, hwaccel: str = "") -> str:
    """Save a single still at time t, tone-mapped to SDR when the source is HDR."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    g, pre, post = hdr_to_sdr(info)
    vf = list(pre)
    if scale_width:
        vf.append(f"scale={scale_width}:-2")
    vf += post
    cmd = ["ffmpeg", "-v", "error", "-y"] + g
    if hwaccel and not (g or post):
        cmd += ["-hwaccel", hwaccel]
    cmd += ["-ss", f"{t:.3f}", "-i", info.path, "-frames:v", "1"]
    if vf:
        cmd += ["-vf", ",".join(vf)]
    cmd += ["-q:v", "2", out_path]
    subprocess.run(cmd, check=True, capture_output=True)
    return out_path


def extract_crop(info, t: float, bbox, out_path: str, *,
                 margin: float = 0.25, out_width: int = 512,
                 frame_w: int, frame_h: int, hwaccel: str = "") -> str:
    """Crop the bird (bbox in full-res coords) at time t, tone-mapped to SDR."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half_w = (x2 - x1) * (1 + margin) / 2
    half_h = (y2 - y1) * (1 + margin) / 2
    nx1, ny1 = max(0, cx - half_w), max(0, cy - half_h)
    nx2, ny2 = min(frame_w, cx + half_w), min(frame_h, cy + half_h)
    w, h = max(2, int(nx2 - nx1)), max(2, int(ny2 - ny1))
    x, y = int(nx1), int(ny1)
    g, pre, post = hdr_to_sdr(info)
    vf = list(pre) + [f"crop={w}:{h}:{x}:{y}", f"scale={out_width}:-2"] + post
    cmd = ["ffmpeg", "-v", "error", "-y"] + g
    if hwaccel and not (g or post):
        cmd += ["-hwaccel", hwaccel]
    cmd += ["-ss", f"{t:.3f}", "-i", info.path, "-frames:v", "1",
            "-vf", ",".join(vf), "-q:v", "2", out_path]
    subprocess.run(cmd, check=True, capture_output=True)
    return out_path
