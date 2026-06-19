"""Stage 4a — compute the crop window that follows the bird's head.

Given the detector's bbox track, produce a per-output-frame crop top-left path:
pick the largest window of the chosen output aspect that fits the source (times
an optional punch-in zoom), center it on the bird's *head* (a point anchored a
fraction `head_anchor` down from the top of the bbox — birds are usually framed
by the head, not the body/tail), then smooth with an EMA + max-pan clamp.

Note: head height is approximated from the bbox; for tall full-height crops the
vertical move is clamped out so only horizontal head-centering applies. True
per-frame head keypoints would need a bird-pose model (a possible upgrade).
"""
from __future__ import annotations

import bisect

import cv2
import numpy as np


def head_from_mask(gray, mask):
    """Estimate the bird's head/eye (x,y) from its segmentation mask + gray image.

    The bird's two body extremities are its head and tail. We pick the head end by
    combining two cues that are individually unreliable but complementary: it is the
    *bulkier* end (tails taper thin) AND the end with the strongest dark/high-contrast
    spot (the eye). The returned point is that darkest spot — i.e. the eye itself.
    """
    ys, xs = np.where(mask)
    if len(xs) < 30:
        return None
    pts = np.column_stack([xs, ys]).astype(np.float32)
    mean = pts.mean(0)
    d = pts - mean
    evals, evecs = np.linalg.eigh((d.T @ d) / len(d))
    axis = evecs[:, -1]                                   # principal (head-to-tail) axis
    proj = d @ axis
    mx, mn = float(proj.max()), float(proj.min())
    if mx - mn < 8:
        return float(mean[0]), float(mean[1])
    g = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 2)
    H, W = mask.shape
    R = 0.22 * (mx - mn)
    feats = []
    for e in (mean + axis * mn, mean + axis * mx):
        x0, x1 = max(0, int(e[0] - R)), min(W, int(e[0] + R) + 1)
        y0, y1 = max(0, int(e[1] - R)), min(H, int(e[1] + R) + 1)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        win = ((xx - e[0]) ** 2 + (yy - e[1]) ** 2 < R * R) & mask[y0:y1, x0:x1]
        n = int(win.sum())
        if n < 10:
            feats.append((0, 0.0, (float(e[0]), float(e[1]))))
            continue
        sg = g[y0:y1, x0:x1]
        iy, ix = np.unravel_index(int(np.argmin(np.where(win, sg, 1e9))), sg.shape)
        feats.append((n, float(sg[win].mean() - sg[win].min()), (float(x0 + ix), float(y0 + iy))))
    bsum = feats[0][0] + feats[1][0] + 1e-6
    csum = feats[0][1] + feats[1][1] + 1e-6
    s0 = 0.4 * (feats[0][0] / bsum) + 0.6 * (feats[0][1] / csum)
    s1 = 0.4 * (feats[1][0] / bsum) + 0.6 * (feats[1][1] / csum)
    return feats[0][2] if s0 > s1 else feats[1][2]


def target_crop_size(W: int, H: int, ar: float, zoom: float = 1.0) -> tuple[float, float]:
    """Largest width:height = `ar` rectangle fitting in WxH, scaled by `zoom`."""
    cw = H * ar
    if cw <= W:
        cw, ch = cw, float(H)
    else:
        cw, ch = float(W), W / ar
    return min(cw * zoom, W), min(ch * zoom, H)


def _head_fn(track: list, anchor: float):
    ts = [d["t"] for d in track]
    hx, hy = [], []
    for d in track:
        h = d.get("head")                                # detected head/eye (seg model)
        if h:
            hx.append(h[0])
            hy.append(h[1])
        else:                                            # fallback: anchor near top of bbox
            x1, y1, x2, y2 = d["bbox"]
            hx.append((x1 + x2) / 2)
            hy.append(y1 + anchor * (y2 - y1))

    def f(t: float) -> tuple[float, float]:
        if t <= ts[0]:
            return hx[0], hy[0]
        if t >= ts[-1]:
            return hx[-1], hy[-1]
        i = bisect.bisect_right(ts, t)
        t0, t1 = ts[i - 1], ts[i]
        a = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
        return hx[i - 1] + a * (hx[i] - hx[i - 1]), hy[i - 1] + a * (hy[i] - hy[i - 1])

    return f


def compute_path(seg, W: int, H: int, render_cfg: dict, zoom: float = 1.0) -> dict:
    speed = render_cfg["speed"]
    out_fps = render_cfg["fps"]
    ar = render_cfg["width"] / render_cfg["height"]      # output aspect (9:16 or 1080:1560)
    cw, ch = target_crop_size(W, H, ar, zoom)
    anchor = render_cfg.get("head_anchor", 0.30)

    n = max(1, int(round((seg.end - seg.start) / speed * out_fps)))
    track = seg.track or []
    f = _head_fn(track, anchor) if (track and render_cfg.get("tracking", True)) else None
    sm = render_cfg["track_smooth"]
    maxpan = render_cfg["max_pan_px_per_frame"]

    xs: list[float] = []
    ys: list[float] = []
    px = py = None
    for i in range(n):
        src_t = seg.start + (i / out_fps) * speed
        cx, cy = f(src_t) if f else (W / 2.0, H / 2.0)
        tx = min(max(0.0, cx - cw / 2.0), W - cw)
        ty = min(max(0.0, cy - ch / 2.0), H - ch)
        if px is None:
            sx, sy = tx, ty
        else:
            sx = px + sm * (tx - px)
            sy = py + sm * (ty - py)
            sx = px + max(-maxpan, min(maxpan, sx - px))
            sy = py + max(-maxpan, min(maxpan, sy - py))
        sx = min(max(0.0, sx), W - cw)
        sy = min(max(0.0, sy), H - ch)
        px, py = sx, sy
        xs.append(sx)
        ys.append(sy)

    return {"crop_w": cw, "crop_h": ch, "n": n, "xs": xs, "ys": ys}
