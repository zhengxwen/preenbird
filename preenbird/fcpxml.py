"""Stage 5 — export selected segments as a Final Cut Pro project (FCPXML).

Each selected segment becomes its own vertical <project>: the source clip trimmed
to its in/out point, spatially conformed to FILL the 9:16 (or 1080x1560) frame, with
a keyframed Transform 'position' that follows the bird's head (from the detection
track) — so the reframe is fully editable, not baked. Clips export at normal speed;
apply a 50% retime in Final Cut for the slow-mo (the head keyframes stretch with it).
"""
from __future__ import annotations

import os
from urllib.parse import quote
from xml.sax.saxutils import escape


def _rt(t: float, fps: float) -> str:
    den = int(round(fps)) * 100
    return f"{int(round(t * fps)) * 100}/{den}s"


def _fd(fps: float) -> str:
    return f"100/{int(round(fps)) * 100}s"


def _ema(vals, a=0.3):
    out, p = [], None
    for v in vals:
        p = v if p is None else p + a * (v - p)
        out.append(p)
    return out


def _head_xy(d, anchor=0.30):
    h = d.get("head")
    if h:
        return h[0], h[1]
    x1, y1, x2, y2 = d["bbox"]
    return (x1 + x2) / 2, y1 + anchor * (y2 - y1)


def generate_fcpxml(info, items, out_path, render_cfg, zoom=1.0):
    """items: list of {seg, start, end, name}. Writes one FCPXML with one project each."""
    out_w, out_h = render_cfg["width"], render_cfg["height"]
    fps = info.fps
    W, H = info.width, info.height
    s = max(out_w / W, out_h / H) / (zoom or 1.0)        # FILL scale (zoom punches in)
    slack_x = max(0.0, W * s - out_w)
    slack_y = max(0.0, H * s - out_h)
    anchor = render_cfg.get("head_anchor", 0.30)

    url = "file://" + quote(os.path.abspath(info.path))
    R = [
        f'<format id="r1" name="Preen {out_w}x{out_h}" frameDuration="{_fd(fps)}" '
        f'width="{out_w}" height="{out_h}" colorSpace="1-1-1 (Rec. 709)"/>',
        f'<format id="r2" name="src {W}x{H}" frameDuration="{_fd(fps)}" width="{W}" height="{H}"/>',
        f'<asset id="r3" name="{escape(os.path.basename(info.path))}" start="0s" hasVideo="1" '
        f'hasAudio="{1 if info.has_audio else 0}" videoSources="1" format="r2" '
        f'duration="{_rt(info.duration, fps)}"><media-rep kind="original-media" src="{url}"/></asset>',
    ]

    projects = []
    for it in items:
        seg = it["seg"]
        src_in = float(it.get("start", seg.start))
        src_out = float(it.get("end", seg.end))
        dur = max(1.0 / fps, src_out - src_in)
        name = escape(str(it.get("name") or seg.species or "bird"))
        samples = [d for d in (seg.track or []) if src_in - 0.05 <= d["t"] <= src_out + 0.05] \
            or (seg.track or [])
        ts, xs, ys = [], [], []
        for d in samples:
            hx, hy = _head_xy(d, anchor)
            px = max(-slack_x / 2, min(slack_x / 2, (W / 2 - hx) * s))
            py = max(-slack_y / 2, min(slack_y / 2, -(H / 2 - hy) * s))   # FCP +Y is up
            ts.append(max(0.0, d["t"] - src_in))
            xs.append(px)
            ys.append(py)
        xs, ys = _ema(xs), _ema(ys)
        kf = "".join(f'<keyframe time="{_rt(t, fps)}" value="{x:.1f} {y:.1f}"/>'
                     for t, x, y in zip(ts, xs, ys)) or '<keyframe time="0s" value="0 0"/>'
        clip = (
            f'<asset-clip ref="r3" offset="0s" name="{name}" start="{_rt(src_in, fps)}" '
            f'duration="{_rt(dur, fps)}" format="r2" tcFormat="NDF">'
            f'<adjust-conform type="fill"/>'
            f'<adjust-transform><param name="position">'
            f'<keyframeAnimation>{kf}</keyframeAnimation></param></adjust-transform>'
            f'</asset-clip>')
        projects.append(
            f'<project name="{name} 9x16"><sequence format="r1" duration="{_rt(dur, fps)}" '
            f'tcStart="0s" tcFormat="NDF" audioLayout="stereo" audioRate="48k">'
            f'<spine>{clip}</spine></sequence></project>')

    xml = ('<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE fcpxml>\n'
           '<fcpxml version="1.9">\n<resources>\n' + "\n".join(R) + "\n</resources>\n"
           '<library><event name="Preen">\n' + "\n".join(projects) +
           "\n</event></library>\n</fcpxml>\n")
    with open(out_path, "w") as f:
        f.write(xml)
    return out_path
