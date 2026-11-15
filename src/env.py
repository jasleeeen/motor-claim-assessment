from __future__ import annotations

import os
from pathlib import Path

from src.paths import ROOT


def parse_env(text: str) -> dict[str, str]: 
    out: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, sep, value = line.partition("=")
        if not sep:
            continue         
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        out[key] = value
    return out


def load_env(path: str | Path | None = None, *, override: bool = False) -> dict[str, str]:
    if path is None and os.environ.get("PYTEST_CURRENT_TEST"):
        return {}

    env_file = Path(path) if path is not None else ROOT / ".env"
    try:
        text = env_file.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return {}

    applied = {}
    for key, value in parse_env(text).items():
        if not value:
            continue
        if override or not os.environ.get(key):
            os.environ[key] = value
            applied[key] = value
    return applied
