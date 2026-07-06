# Preen

🐦 **Preen** — auto-edit fixed-camera bird videos into 9:16 YouTube Shorts: AI detection, subject tracking, slow-mo & music.

> The PyPI package, repository, and CLI are named `preenbird`; the tool/brand is **Preen**.

Point a fixed camera at a feeder, let it roll, and Preen finds the bird, throws away the empty footage, ranks what's worth posting, and cuts a vertical Short that tracks the bird with 0.5× slow-mo and backing music.

## Pipeline

```
preen detect <video…>   ① ffmpeg-sampled frames → YOLO11 bird detection → candidate segments
preen score  <video…>   ② crop the bird → local Ollama vision model → species + interest score
preen scan   <video…>   ①+② in one pass
preen review            ③ local web UI: rank, preview, trim in/out, pick what to publish
preen render <video…>   ④ tracking 9:16 crop + 0.5× slow-mo + music → 1080×1920 (.mp4)
preen export <video…>   ④′ Final Cut Pro project (.fcpxml) instead of a rendered file
```

## Install

```bash
cd preenbird
python3 -m venv --system-site-packages .venv     # reuse system torch (MPS / CUDA)
.venv/bin/pip install -e .
```

External system dependencies (not pip):

- **ffmpeg + ffprobe** on `PATH`
- a running **Ollama** server with a vision model, e.g. `ollama pull gemma3:12b`
- YOLO weights (`yolo11m-seg.pt`) — auto-downloaded by Ultralytics on first run if absent

Optionally symlink the launcher so `preen` works from any directory:

```bash
ln -s "$PWD/bin/preen" /usr/local/bin/preen
```

## Usage

```bash
preen scan ~/Footage/feeder_2026-06-19.mov     # detect + score
preen review                                   # open the web UI at http://127.0.0.1:8765
preen render ~/Footage/feeder_2026-06-19.mov --top 3
```

Copy `config.example.yaml` to `config.yaml` to override defaults (work dir, music library, model names, render settings, watermark).

## Architecture

- **Backend** — Python (`preenbird` package) + FastAPI. Detection via Ultralytics YOLO11, encoding via ffmpeg (VideoToolbox on macOS / NVENC on the RTX 5090 box), scoring via a local Ollama vision model.
- **Frontend** — plain HTML/CSS/JS in `preenbird/review/static/` (`index.html` · `styles.css` · `app.js`), served by FastAPI. No build step.

## License

[AGPL-3.0-or-later](LICENSE). Preen depends on [Ultralytics YOLO](https://github.com/ultralytics/ultralytics), which is AGPL-3.0; the whole project is therefore distributed under AGPL-3.0. Commercial use without open-sourcing requires an Ultralytics Enterprise License (or swapping the detector for a non-AGPL model).
