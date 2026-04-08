from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

try:
    import yaml
except ImportError as exc:  # pragma: no cover - explicit dependency failure
    raise ImportError(
        "PyYAML is required to load config.yaml. Install with `pip install pyyaml`."
    ) from exc

_CONFIG_CACHE: Optional[Dict[str, Any]] = None


def load_config(path: Optional[Path] = None) -> Dict[str, Any]:
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None and path is None:
        return _CONFIG_CACHE

    config_path = path or (Path(__file__).resolve().parents[1] / "config.yaml")
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    if path is None:
        _CONFIG_CACHE = config
    return config
