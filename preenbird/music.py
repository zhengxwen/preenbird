"""Pick a backing track from the user's library folder."""
from __future__ import annotations

import os

AUDIO_EXTS = {".mp3", ".m4a", ".aac", ".wav", ".flac", ".ogg", ".aiff", ".aif"}


def list_tracks(music_dir: str) -> list[str]:
    """All audio files under music_dir, recursively (subfolders included)."""
    if not music_dir or not os.path.isdir(music_dir):
        return []
    out = []
    for root, _dirs, files in os.walk(music_dir):
        for f in files:
            if os.path.splitext(f)[1].lower() in AUDIO_EXTS:
                out.append(os.path.join(root, f))
    return sorted(out)


def pick_track(music_dir: str, index: int = 0) -> str | None:
    """Return a track path (looped/trimmed to fit happens at render time)."""
    tracks = list_tracks(music_dir)
    if not tracks:
        return None
    return tracks[index % len(tracks)]
