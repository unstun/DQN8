"""JSON configuration loading and argparse integration.

Provides
--------
- load_json()              Read a JSON file into a dict.
- resolve_config_path()    Resolve --config / --profile to a Path under configs/.
- select_section()         Extract the "train" or "infer" sub-dict from a config.
- apply_config_defaults()  Merge config dict values into an argparse parser's defaults,
                           with type coercion matching each argument's declared type.
- parser_defaults()        Dump all argparse defaults as a JSON-compatible dict.

Config resolution order
-----------------------
1. --profile <name>  → configs/<name>.json
2. --config <path>   → literal path (with fallback to configs/ prefix)
3. (neither)         → configs/config.json if exists, else None
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _json_compatible(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_json_compatible(v) for v in value]
    if isinstance(value, list):
        return [_json_compatible(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_compatible(v) for k, v in value.items()}
    return value


def load_json(path: Path) -> dict[str, Any]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Config must be a JSON object at top-level: {path}")
    return raw


def resolve_config_path(
    *,
    config: Path | None,
    profile: str | None,
    default_path: Path = Path("configs/config.json"),
    profiles_dir: Path = Path("configs"),
) -> Path | None:
    if config is not None and profile is not None:
        raise ValueError("Use only one of --config or --profile.")

    if profile is not None:
        p = Path(str(profile).strip())
        candidates: list[Path] = []
        if p.is_absolute():
            candidates.append(p)
            if p.suffix == "":
                candidates.append(p.with_suffix(".json"))
        else:
            candidates.append(profiles_dir / p)
            if p.suffix == "":
                candidates.append((profiles_dir / p).with_suffix(".json"))

        for cand in candidates:
            if cand.is_file():
                return cand
        raise FileNotFoundError(f"Config profile not found: {profile!r} (looked for {', '.join(map(str, candidates))})")

    if config is not None:
        p = Path(config)
        candidates: list[Path] = [p]
        if p.suffix == "":
            candidates.append(p.with_suffix(".json"))
        # Also allow passing a bare name (or bare filename) and looking under configs/.
        candidates.append(profiles_dir / p)
        if p.suffix == "":
            candidates.append((profiles_dir / p).with_suffix(".json"))

        for cand in candidates:
            if cand.is_file():
                return cand
        raise FileNotFoundError(f"Config file not found: {str(config)!r} (looked for {', '.join(map(str, candidates))})")

    if default_path.is_file():
        return default_path
    return None


def _unwrap_args_payload(cfg: dict[str, Any]) -> dict[str, Any]:
    # Support re-using runs/<...>/configs/run.json (it stores args under "args").
    args = cfg.get("args")
    if isinstance(args, dict):
        return args
    return cfg


def select_section(cfg: dict[str, Any], *, section: str) -> dict[str, Any]:
    cfg = _unwrap_args_payload(cfg)
    sect = cfg.get(section)
    if isinstance(sect, dict):
        return sect
    return cfg


def parser_defaults(parser: argparse.ArgumentParser, *, exclude: set[str] | None = None) -> dict[str, object]:
    exclude = set() if exclude is None else set(exclude)
    out: dict[str, object] = {}
    for act in parser._actions:
        dest = getattr(act, "dest", None)
        if not dest or dest in ("help",) or dest in exclude:
            continue
        if getattr(act, "default", argparse.SUPPRESS) is argparse.SUPPRESS:
            continue
        out[str(dest)] = _json_compatible(getattr(act, "default"))
    return out


def _action_by_dest(parser: argparse.ArgumentParser) -> dict[str, argparse.Action]:
    actions: dict[str, argparse.Action] = {}
    for act in parser._actions:
        dest = getattr(act, "dest", None)
        if not dest or dest == "help":
            continue
        actions[str(dest)] = act
    return actions


def _coerce_list(value: object) -> list[object]:
    if value is None:
        return []
    if isinstance(value, str):
        parts = [p for p in value.split() if p]
        return [*parts]
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _coerce_action_value(action: argparse.Action, value: object) -> object:
    if value is None:
        return None

    # nargs="*" / "+" style args.
    nargs = getattr(action, "nargs", None)
    if nargs in ("*", "+") or isinstance(value, list):
        items = _coerce_list(value)
        if action.type is None:
            return items
        return [action.type(v) if v is not None else None for v in items]

    if action.type is None:
        return value
    return action.type(value)


def apply_config_defaults(
    parser: argparse.ArgumentParser,
    cfg: dict[str, Any],
    *,
    strict: bool = True,
    allow_unknown_prefixes: tuple[str, ...] = ("_",),
) -> None:
    actions = _action_by_dest(parser)

    unknown = []
    coerced: dict[str, object] = {}
    for k, v in cfg.items():
        key = str(k)
        if any(key.startswith(p) for p in allow_unknown_prefixes):
            continue
        act = actions.get(key)
        if act is None:
            unknown.append(key)
            continue
        coerced[key] = _coerce_action_value(act, v)

    if strict and unknown:
        unknown_s = ", ".join(sorted(set(unknown)))
        raise ValueError(f"Unknown config keys for this command: {unknown_s}")

    if coerced:
        parser.set_defaults(**coerced)
