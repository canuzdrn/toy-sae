"""Small helpers for command-line scripts."""

from __future__ import annotations

import json
from pathlib import Path
import shlex
import sys
from typing import Any


def to_jsonable(value: Any) -> Any:
    """Convert common script values into JSON-serializable objects."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            pass
    return value


def args_to_dict(args: Any, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return an argparse namespace as a JSON-friendly dictionary."""
    config = {
        key: to_jsonable(value)
        for key, value in vars(args).items()
    }
    if extra:
        config.update({key: to_jsonable(value) for key, value in extra.items()})
    return config


def command_string(argv: list[str] | None = None) -> str:
    """Return a shell-escaped command string for provenance metadata."""
    if argv is None:
        argv = sys.argv
    return " ".join(shlex.quote(str(part)) for part in argv)


def command_text(command: list[Any]) -> str:
    """Return a shell-escaped command string for a subprocess command list."""
    return " ".join(shlex.quote(str(part)) for part in command)


def project_path(path: str | Path, project_root: Path) -> Path:
    """Resolve a possibly relative path against the project root."""
    path = Path(path)
    if path.is_absolute():
        return path
    return project_root / path


def save_json(path: str | Path, payload: Any, *, sort_keys: bool = False) -> None:
    """Write JSON with project-standard indentation and a trailing newline."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=sort_keys) + "\n")
