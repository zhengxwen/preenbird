"""Preen — auto-edit fixed-camera bird footage into YouTube Shorts.

Pipeline:
  1. detect  — ffmpeg-sampled frames -> YOLO bird detection -> candidate segments
  2. score   — crop the bird -> local Ollama vision model -> species + interest score
  3. review  — local web UI to pick/trim the segments to publish
  4. render  — tracking 9:16 crop + 0.5x slow-mo + music -> 1080x1920 Short (+ FCPXML)
"""

__version__ = "0.1.0"
