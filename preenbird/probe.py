"""Thin ffprobe wrapper to read real clip specs.

Reports *display* width/height (after applying the rotation side-data), because
ffmpeg auto-rotates frames on decode — so every downstream coordinate (detection
bbox, crops, reframe) must be in display orientation to line up.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass


@dataclass
class VideoInfo:
    path: str
    width: int            # display orientation
    height: int           # display orientation
    fps: float
    duration: float
    codec: str
    has_audio: bool
    rotation: int = 0
    hdr: bool = False
    transfer: str = ""        # color_transfer, e.g. smpte2084 (PQ) / arib-std-b67 (HLG)

    def as_dict(self) -> dict:
        return asdict(self)


def _parse_rate(rate: str) -> float:
    try:
        num, den = rate.split("/")
        den = float(den)
        return float(num) / den if den else 0.0
    except (ValueError, ZeroDivisionError):
        return 0.0


def _rotation(video: dict) -> int:
    for sd in video.get("side_data_list", []):
        if "rotation" in sd:
            try:
                return int(round(float(sd["rotation"])))
            except (TypeError, ValueError):
                pass
    rot = video.get("tags", {}).get("rotate")
    if rot:
        try:
            return int(rot)
        except ValueError:
            pass
    return 0


def _is_hdr(video: dict) -> bool:
    if video.get("color_transfer") in ("smpte2084", "arib-std-b67"):
        return True
    return any(sd.get("side_data_type", "").startswith("DOVI")
               for sd in video.get("side_data_list", []))


def probe(path: str) -> VideoInfo:
    cmd = ["ffprobe", "-v", "error", "-print_format", "json",
           "-show_streams", "-show_format", path]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    data = json.loads(out)

    video = next(s for s in data["streams"] if s["codec_type"] == "video")
    audio = next((s for s in data["streams"] if s["codec_type"] == "audio"), None)

    fps = _parse_rate(video.get("avg_frame_rate", "0/1")) or \
        _parse_rate(video.get("r_frame_rate", "30/1")) or 30.0
    duration = float(data["format"].get("duration") or video.get("duration") or 0.0)

    cw, ch = int(video["width"]), int(video["height"])
    rot = _rotation(video)
    if abs(rot) % 180 == 90:        # portrait: swap to display orientation
        dw, dh = ch, cw
    else:
        dw, dh = cw, ch

    return VideoInfo(
        path=path,
        width=dw,
        height=dh,
        fps=round(fps, 3),
        duration=round(duration, 3),
        codec=video.get("codec_name", ""),
        has_audio=audio is not None,
        rotation=rot,
        hdr=_is_hdr(video),
        transfer=video.get("color_transfer", ""),
    )
