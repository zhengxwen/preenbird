"""Configuration: built-in defaults deep-merged with an optional config.yaml."""
from __future__ import annotations

import copy
import os

try:
    import yaml
except ImportError:  # pyyaml ships with ultralytics, so this should not happen
    yaml = None

DEFAULTS = {
    "paths": {
        "music_dir": "~/Music/Preen",
        "work_dir": "./data",
    },
    "models": {
        "detector": "yolo11m-seg.pt",   # seg model: masks enable head tracking
        "vlm": "gemma4:12b-it-qat",
        "vlm_high": "gemma4:26b-a4b-it-qat",
    },
    "detect": {
        "sample_fps": 2.0,
        "scale_width": 1280,
        "conf": 0.30,
        "min_bird_frac": 0.0025,
        "merge_gap_s": 1.5,
        "pad_lead_s": 0.6,
        "pad_tail_s": 1.0,
        "min_segment_s": 5.0,
        "batch": 16,
        "device": "mps",
        "hwaccel": "videotoolbox",
    },
    "score": {
        "frames_per_segment": 5,
        "highlight_s": 14.0,
        "crop_margin": 0.25,
        "crop_width": 512,
        "preference": (
            "These are stills from a backyard fixed-camera bird feeder, curated for "
            "YouTube Shorts. Favor colorful, striking, or uncommon birds and interesting "
            "behavior. Treat very common 'little brown jobs' (house sparrows, starlings, "
            "pigeons, grackles) as low interest unless the shot is exceptional."
        ),
        "favor": ["woodpecker", "cardinal", "blue jay", "bluebird", "goldfinch",
                  "oriole", "nuthatch", "chickadee", "warbler", "hummingbird"],
        "demote": ["house sparrow", "sparrow", "european starling", "starling",
                   "pigeon", "rock dove", "common grackle", "grackle"],
    },
    "render": {
        "width": 1080,
        "height": 1920,
        "fps": "auto",          # source_fps * speed for 1:1 frames (60->30, 50->25); or an int
        "speed": 0.5,
        "tracking": True,
        "track_smooth": 0.12,
        "max_pan_px_per_frame": 18,
        "codec": "hevc_vt",             # hevc_vt | x265 | x264 | av1
        "quality": "high",              # high | medium | low
        "bitrate": "8M",                # used by hevc_vt
        "music_vol": 0.85,              # 0..1 backing-music level
        "natural_vol": 0.2,             # 0..1 original-audio level (mixed under music)
        "music_lufs": -16,
        "keep_natural_audio": True,
        "natural_audio_lufs": -26,
        "fade_s": 0.4,
        "caption_species": True,        # overlay species name (optional)
        "head_anchor": 0.30,            # tracking focus: fraction down the bbox (~head height)
        "tonemap": "hable",             # PQ HDR tone-map operator
        "hdr_contrast": 1.06,           # post-tonemap contrast (fixes grey/washed HDR)
        "hdr_saturation": 1.15,         # post-tonemap saturation
        "watermark": {
            "enabled": False,
            "image": "",                # path to a PNG/logo (takes priority over text)
            "text": "",                 # or a text watermark (e.g. channel handle)
            "position": "br",           # tl | tr | bl | br
            "opacity": 0.85,
            "scale": 0.14,              # logo width as a fraction of video width
        },
    },
}


def deep_merge(base: dict, over: dict | None) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _expand_paths(d):
    for k, v in d.items():
        if isinstance(v, dict):
            _expand_paths(v)
        elif isinstance(v, str) and v.startswith("~"):
            d[k] = os.path.expanduser(v)
    return d


def load_config(path: str | None = None) -> dict:
    """Load defaults, optionally overlaying a YAML file. Returns a plain dict."""
    cfg = copy.deepcopy(DEFAULTS)
    if path and os.path.exists(path):
        if yaml is None:
            raise RuntimeError("pyyaml not available to read config file")
        with open(path) as f:
            user = yaml.safe_load(f) or {}
        cfg = deep_merge(cfg, user)
    elif path:
        raise FileNotFoundError(path)
    return _expand_paths(cfg)
