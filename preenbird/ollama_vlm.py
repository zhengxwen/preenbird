"""Minimal client for a local Ollama vision model (structured JSON output)."""
from __future__ import annotations

import base64
import json

import requests

OLLAMA_URL = "http://localhost:11434"


def _repair_json(text: str) -> dict:
    """Salvage a truncated/garnished JSON object by closing open strings/braces."""
    a = text.find("{")
    if a < 0:
        raise ValueError("no JSON object in response")
    out, depth, in_str, esc = [], 0, False, False
    for ch in text[a:]:
        out.append(ch)
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                break
    if in_str:
        out.append('"')
    out.append("}" * max(0, depth))
    return json.loads("".join(out))


def vlm_json(model: str, prompt: str, image_path: str,
             schema: dict | None = None, timeout: int = 180) -> dict:
    """Send an image + prompt to Ollama and parse a JSON response.

    If `schema` is given it is passed as Ollama's `format` for constrained
    decoding; otherwise we request free-form JSON.
    """
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()

    payload = {
        "model": model,
        "prompt": prompt,
        "images": [b64],
        "stream": False,
        "options": {"temperature": 0.2, "num_predict": 512},
        "format": schema if schema else "json",
    }
    r = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=timeout)
    r.raise_for_status()
    text = r.json().get("response", "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return _repair_json(text)


def health() -> tuple[bool, str]:
    """Return (ok, message) for the Ollama server."""
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        r.raise_for_status()
        models = [m["name"] for m in r.json().get("models", [])]
        return True, f"{len(models)} models available"
    except Exception as e:  # noqa: BLE001
        return False, str(e)
