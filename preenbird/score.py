"""Stage 2 — judge species + how interesting/beautiful each segment is.

For each candidate segment we crop the bird from its sharpest sampled frame and
ask the local Ollama vision model for a structured judgement. The model's
beauty/interestingness ratings are combined with the user's favor/demote
preferences and the detector's own size/confidence signals into a single
0-100 `publish_score` used to rank the review list.
"""
from __future__ import annotations

import os
import re

from .detect import Segment
from .media import extract_crop
from .ollama_vlm import vlm_json
from .probe import VideoInfo

BIRD_SCHEMA = {
    "type": "object",
    "properties": {
        "is_bird": {"type": "boolean"},
        "common_name": {"type": "string"},
        "species_guess": {"type": "string"},
        "family": {"type": "string"},
        "id_confidence": {"type": "number"},
        "beauty": {"type": "integer"},
        "interestingness": {"type": "integer"},
        "behavior": {"type": "string"},
        "common_lbj": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["is_bird", "common_name", "id_confidence", "beauty",
                 "interestingness", "common_lbj", "reason"],
}


def _clean_text(s, limit=140):
    """Strip code fences / non-ASCII / repeated-token garbage from flaky VLM JSON."""
    if not isinstance(s, str):
        return s
    s = re.sub(r"`{1,}\w*", "", s)
    s = re.sub(r"[^\x20-\x7E]+", " ", s)
    s = re.sub(r"(.{1,4}?)\1{3,}", r"\1", s)        # collapse any short repeated run (|/|/, ceserce)
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w.!?)\]]+$", "", s).strip()      # trim trailing junk
    return s[:limit]


def build_prompt(score_cfg: dict) -> str:
    favor = ", ".join(score_cfg["favor"])
    demote = ", ".join(score_cfg["demote"])
    return (
        f"{score_cfg['preference']}\n\n"
        "Identify the bird in this cropped photo from a backyard feeder camera.\n"
        f"More interesting (favor): {favor}.\n"
        f"Less interesting / common (demote): {demote}.\n\n"
        "Return JSON with: is_bird (bool), common_name, species_guess, family, "
        "id_confidence (0-1), beauty (1-10), interestingness (1-10), "
        "behavior (a few words), common_lbj (true if a drab/common species), "
        "reason (one short sentence). If it is not actually a bird, set is_bird=false "
        "and the ratings to 0."
    )


def compute_publish_score(seg: Segment, score_cfg: dict) -> float:
    beauty = seg.beauty or 0
    interest = seg.interestingness or 0
    score = 5.0 * interest + 3.0 * beauty          # 0..80 from the model

    name = f"{seg.species or ''} {seg.species_guess or ''} {seg.family or ''}".lower()
    if any(k in name for k in score_cfg["favor"]):
        score += 12
    if seg.common_lbj or any(k in name for k in score_cfg["demote"]):
        score -= 18

    score += min(10.0, (seg.max_area_frac or 0) * 100)   # reward a big, clear bird
    score += 5.0 * (seg.peak_conf or 0)
    return round(max(0.0, min(100.0, score)), 1)


def _pick_frames(track, n):
    """Pick up to n frames spread evenly across the track, always incl. the sharpest."""
    best = max(track, key=lambda d: d["conf"])
    if n <= 1 or len(track) <= 1:
        return [best]
    n = min(n, len(track))
    idxs = sorted({round(i * (len(track) - 1) / (n - 1)) for i in range(n)})
    frames = [track[i] for i in idxs]
    if best not in frames:
        frames[len(frames) // 2] = best
    return frames


def score_segment(info: VideoInfo, seg: Segment, score_cfg: dict, model: str,
                  work_dir: str, hwaccel: str = "", n_frames=None, progress=None) -> Segment:
    """Score a segment from N frames (default score_cfg.frames_per_segment): each is
    cropped to the bird and judged by the VLM; ratings are averaged and the species is
    taken from the most-confident frame. `progress(j, total)` is called per frame."""
    if not seg.track:
        return seg
    seg.vlm_error = None
    if n_frames is None:
        n_frames = score_cfg.get("frames_per_segment", 5)
    frames = _pick_frames(seg.track, n_frames)
    best = max(seg.track, key=lambda d: d["conf"])
    prompt = build_prompt(score_cfg)
    crops_dir = os.path.join(work_dir, "crops")
    results, seg_crop = [], None
    for j, d in enumerate(frames):
        cp = os.path.join(crops_dir, f"seg_{seg.idx:03d}_{j}.jpg")
        try:
            extract_crop(info, d["t"], d["bbox"], cp, margin=score_cfg["crop_margin"],
                         out_width=score_cfg["crop_width"], frame_w=info.width,
                         frame_h=info.height, hwaccel=hwaccel)
            results.append(vlm_json(model, prompt, cp, schema=BIRD_SCHEMA))
            if d is best:
                seg_crop = cp
        except Exception as e:  # noqa: BLE001
            seg.vlm_error = str(e)
        if progress:
            progress(j + 1, len(frames))
    if not results:
        return seg
    seg.vlm_error = None

    def avg(key):
        v = [r[key] for r in results if isinstance(r.get(key), (int, float))]
        return round(sum(v) / len(v)) if v else None

    seg.beauty = avg("beauty")
    seg.interestingness = avg("interestingness")
    bestr = max(results, key=lambda r: r.get("id_confidence") or 0)    # most-confident ID
    seg.species = _clean_text(bestr.get("common_name"), 40)
    seg.species_guess = _clean_text(bestr.get("species_guess"), 40)
    seg.family = _clean_text(bestr.get("family"), 40)
    seg.id_confidence = bestr.get("id_confidence")
    seg.behavior = _clean_text(bestr.get("behavior"), 80)
    seg.reason = _clean_text(bestr.get("reason"), 140)
    lbj = [bool(r.get("common_lbj")) for r in results]
    seg.common_lbj = sum(lbj) > len(lbj) / 2                            # majority vote
    seg.crop = os.path.relpath(seg_crop or os.path.join(crops_dir, f"seg_{seg.idx:03d}_0.jpg"), work_dir)
    seg.publish_score = compute_publish_score(seg, score_cfg)
    return seg
