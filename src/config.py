"""Configuration loading and project-relative path resolution."""

from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load YAML configuration and attach resolved project paths."""
    config_path = Path(path).expanduser() if path else DEFAULT_CONFIG_PATH
    if not config_path.is_absolute():
        config_path = (PROJECT_ROOT / config_path).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    config["_project_root"] = PROJECT_ROOT
    config["_config_path"] = config_path
    return config


def resolve_path(config: dict[str, Any], value: str | Path | None) -> Path | None:
    """Resolve an optional path against the project root."""
    if value is None or str(value).strip() == "":
        return None
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (Path(config["_project_root"]) / path).resolve()


def project_path(config: dict[str, Any], key: str) -> Path:
    """Resolve a configured path entry under the ``paths`` section."""
    try:
        value = config["paths"][key]
    except KeyError as error:
        raise KeyError(f"Missing config entry: paths.{key}") from error
    path = resolve_path(config, value)
    if path is None:
        raise ValueError(f"paths.{key} must not be empty")
    return path
