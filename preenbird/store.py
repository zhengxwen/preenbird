"""Per-video working directory: candidates.json read/write."""
from __future__ import annotations

import json
import os

from .detect import Segment
from .probe import VideoInfo


def work_dir_for(cfg: dict, video_path: str) -> str:
    stem = os.path.splitext(os.path.basename(video_path))[0]
    d = os.path.join(cfg["paths"]["work_dir"], stem)
    os.makedirs(d, exist_ok=True)
    return d


def save_candidates(work_dir: str, info: VideoInfo, segments: list[Segment]) -> str:
    path = os.path.join(work_dir, "candidates.json")
    payload = {
        "video": info.as_dict(),
        "segments": [s.to_dict() for s in segments],
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path


def load_candidates(work_dir: str) -> tuple[VideoInfo, list[Segment]]:
    with open(os.path.join(work_dir, "candidates.json")) as f:
        payload = json.load(f)
    info = VideoInfo(**payload["video"])
    segments = [Segment.from_dict(s) for s in payload["segments"]]
    return info, segments
