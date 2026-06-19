"""Stage 1 — find the segments that actually contain a bird.

ffmpeg samples frames at a low fps (cheap), YOLO flags the 'bird' class, and
consecutive hits are merged into padded candidate segments. Output is a list of
Segment objects with a per-sample bbox track (full-res coords) for later scoring
and tracking.
"""
from __future__ import annotations

import os
import subprocess
from dataclasses import asdict, dataclass, field
from typing import Iterator

import cv2
import numpy as np

from .probe import VideoInfo, probe
from .reframe import head_from_mask


@dataclass
class Segment:
    idx: int
    start: float
    end: float
    peak_conf: float
    max_area_frac: float
    n_frames: int
    track: list = field(default_factory=list)   # [{t, bbox:[x1,y1,x2,y2] full-res, conf}]
    # --- filled by stage 2 (score.py) ---
    species: str | None = None
    species_guess: str | None = None
    family: str | None = None
    id_confidence: float | None = None
    beauty: int | None = None
    interestingness: int | None = None
    behavior: str | None = None
    common_lbj: bool | None = None
    reason: str | None = None
    crop: str | None = None
    thumb: str | None = None
    publish_score: float | None = None
    vlm_error: str | None = None

    @property
    def duration(self) -> float:
        return round(self.end - self.start, 3)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["duration"] = self.duration
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Segment":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


def scaled_dims(w: int, h: int, target_w: int) -> tuple[int, int]:
    target_w = min(target_w, w)
    target_w -= target_w % 2
    nh = int(round(h * target_w / w))
    nh -= nh % 2
    return target_w, nh


def iter_frames(info: VideoInfo, sample_fps: float, scale_width: int,
                hwaccel: str = "") -> Iterator[tuple[float, np.ndarray]]:
    """Yield (timestamp, BGR frame) decoded by ffmpeg at `sample_fps`."""
    ow, oh = scaled_dims(info.width, info.height, scale_width)
    cmd = ["ffmpeg", "-v", "error"]
    if hwaccel:
        cmd += ["-hwaccel", hwaccel]
    cmd += ["-i", info.path,
            "-vf", f"fps={sample_fps},scale={ow}:{oh}",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]   # BGR: ultralytics convention
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=10 ** 8)
    frame_bytes = ow * oh * 3
    idx = 0
    try:
        while True:
            buf = proc.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                break
            frame = np.frombuffer(buf, dtype=np.uint8).reshape(oh, ow, 3)
            yield idx / sample_fps, frame
            idx += 1
    finally:
        proc.stdout.close()
        proc.wait()


def _resolve_weights(weights: str) -> str:
    """Resolve a bare weights filename to the copy that ships in the project root.

    When preen is launched from another directory (see bin/preen), a bare
    name like 'yolo11m-seg.pt' would otherwise be downloaded fresh into the caller's
    cwd. If the project already holds that file, use it instead of re-downloading.
    Explicit paths and files present in the cwd are respected as-is.
    """
    if os.path.isabs(weights) or os.path.dirname(weights) or os.path.exists(weights):
        return weights
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # project root
    cand = os.path.join(root, weights)
    return cand if os.path.exists(cand) else weights


def load_detector(weights: str, device: str = "mps"):
    """Return (model, bird_class_index)."""
    from ultralytics import YOLO
    model = YOLO(_resolve_weights(weights))
    bird_ids = [i for i, n in model.names.items() if n == "bird"]
    return model, (bird_ids[0] if bird_ids else 14)


def detect_video(info: VideoInfo, det_cfg: dict, model, bird_class: int,
                 progress=None) -> list[Segment]:
    ow, oh = scaled_dims(info.width, info.height, det_cfg["scale_width"])
    sx, sy = info.width / ow, info.height / oh
    frame_area = float(ow * oh)
    total = max(1, int(round(info.duration * det_cfg["sample_fps"])))

    samples: list[tuple[float, float, tuple | None, float]] = []  # (t, conf, bbox_scaled, area_frac)
    batch, times = [], []

    def flush():
        if not batch:
            return
        results = model.predict(
            batch, classes=[bird_class], conf=det_cfg["conf"],
            imgsz=det_cfg["scale_width"], device=det_cfg.get("device", "mps"),
            retina_masks=True, verbose=False,
        )
        for frame, t, res in zip(batch, times, results):
            boxes = res.boxes
            if boxes is not None and len(boxes):
                confs = boxes.conf.cpu().numpy()
                xyxy = boxes.xyxy.cpu().numpy()
                i = int(confs.argmax())
                x1, y1, x2, y2 = (float(v) for v in xyxy[i])
                area = (x2 - x1) * (y2 - y1) / frame_area
                head = None
                if res.masks is not None and i < len(res.masks.data):
                    mask = res.masks.data[i].cpu().numpy() > 0.5
                    if mask.shape == frame.shape[:2]:
                        head = head_from_mask(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), mask)
                samples.append((t, float(confs[i]), (x1, y1, x2, y2), area, head))
            else:
                samples.append((t, 0.0, None, 0.0, None))
        batch.clear()
        times.clear()

    for t, frame in iter_frames(info, det_cfg["sample_fps"], det_cfg["scale_width"],
                                det_cfg.get("hwaccel", "")):
        batch.append(frame)
        times.append(t)
        if len(batch) >= det_cfg.get("batch", 16):
            flush()
            if progress:
                progress(len(samples), total)
    flush()
    if progress:
        progress(len(samples), total)

    return _group(samples, det_cfg, info.duration, sx, sy)


def _group(samples, det_cfg, duration, sx, sy) -> list[Segment]:
    hits = [s for s in samples
            if s[1] >= det_cfg["conf"] and s[3] >= det_cfg["min_bird_frac"]]
    if not hits:
        return []

    gap = det_cfg["merge_gap_s"]
    groups, cur = [], [hits[0]]
    for s in hits[1:]:
        if s[0] - cur[-1][0] <= gap:
            cur.append(s)
        else:
            groups.append(cur)
            cur = [s]
    groups.append(cur)

    out: list[Segment] = []
    for g in groups:
        start = max(0.0, g[0][0] - det_cfg["pad_lead_s"])
        end = min(duration, g[-1][0] + det_cfg["pad_tail_s"]) if duration else g[-1][0] + det_cfg["pad_tail_s"]
        if end - start < det_cfg["min_segment_s"]:
            continue
        track = []
        for t, conf, bbox, _area, head in g:
            if bbox:
                x1, y1, x2, y2 = bbox
                d = {
                    "t": round(t, 3),
                    "bbox": [round(x1 * sx, 1), round(y1 * sy, 1),
                             round(x2 * sx, 1), round(y2 * sy, 1)],
                    "conf": round(conf, 3),
                }
                if head:
                    d["head"] = [round(head[0] * sx, 1), round(head[1] * sy, 1)]
                track.append(d)
        out.append(Segment(
            idx=len(out),
            start=round(start, 3),
            end=round(end, 3),
            peak_conf=round(max(s[1] for s in g), 3),
            max_area_frac=round(max(s[3] for s in g), 4),
            n_frames=len(g),
            track=track,
        ))
    return out
