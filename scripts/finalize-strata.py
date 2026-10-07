#!/usr/bin/env python3
"""Finalize a Strata run config: resolve path tokens, inject the local API key.

Reads a repo template (configs/strata-iq3_s-*.json, which carries @STRATA_DIR@,
@DATA_DIR@, @GGUF_DIR@ placeholders and a placeholder api_key), turns it into a
runnable config for this install and writes it to --out.

  finalize-strata.py <template.json> --out Strata/strata-iq3_s.json \
      [--strata-dir D] [--data-dir D] [--gguf-dir D] [--key-file F] [--port N]

- the api_key comes from --key-file (default <strata-dir>/api_key.txt); the value is
  never printed. Refuses to write a config that listens on 0.0.0.0 without a key.
- if Strata's setup already wrote its own run config (strata-*.json in the Strata
  folder) for the same GPU layout, its calibrated --pcie-frac / --spec-min-p win
  over the template's (they were measured on this PC).
- aborts if any @PATH token@ survives resolution.

Standard library only.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
ANY_TOKEN_RE = re.compile(r"@[A-Z][A-Z0-9_]*@")


def load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} is not a JSON object")
    return data


def resolve_tokens(obj: Any, tokens: dict[str, str]) -> Any:
    """Recursively replace @NAME@ placeholders in every string value."""
    if isinstance(obj, str):
        for name, value in tokens.items():
            obj = obj.replace("@" + name + "@", value)
        return obj
    if isinstance(obj, list):
        return [resolve_tokens(x, tokens) for x in obj]
    if isinstance(obj, dict):
        return {k: resolve_tokens(v, tokens) for k, v in obj.items()}
    return obj


def gpu_count(cfg: dict[str, Any]) -> int:
    g = cfg.get("gpu", 0)
    return len(g) if isinstance(g, list) else 1


def flag_value(args: list[str], flag: str) -> str | None:
    if flag in args:
        i = args.index(flag)
        if i + 1 < len(args):
            return args[i + 1]
    return None


def setup_run_config(strata_dir: Path, want_gpus: int) -> dict | None:
    """Strata's own run config (written by ./setup.sh in the same folder) for the
    same GPU count, if present — source for per-PC calibrated values."""
    for cand in sorted(strata_dir.glob("strata-*.json")):
        try:
            c = load_json(cand)
        except (OSError, ValueError):
            continue
        if c.get("exe") and isinstance(c.get("args"), list) and gpu_count(c) == want_gpus:
            return c
    return None


def adopt_from_setup(cfg: dict[str, Any], strata_dir: Path) -> None:
    """Per-PC values from Strata's setup-written config win over the template's:
    calibrated --pcie-frac / --spec-min-p, and lib_dirs (the CUDA runtime dirs setup found).

    --pcie-frac is only ever written by ./setup.sh --calibrate, so it is the marker that
    calibration ran; a stock setup config (which carries an uncalibrated
    --spec-min-p 0.5 and no --pcie-frac) is deliberately NOT adopted from, so
    the template's tuned values survive."""
    sc = setup_run_config(strata_dir, gpu_count(cfg))
    if not sc:
        return
    sargs = sc.get("args", [])
    if flag_value(sargs, "--pcie-frac"):
        cal = {}
        for flag in ("--pcie-frac", "--spec-min-p"):
            v = flag_value(sargs, flag)
            if v:
                cal[flag] = v
        if cal:
            cfg["args"] = apply_calibrated(list(cfg.get("args", [])), cal)
            print(f"[finalize] adopting calibrated {cal} from the setup-written config")
    else:
        print("[finalize] setup config is not calibrated (--pcie-frac absent); keeping template values")
    if not cfg.get("lib_dirs") and sc.get("lib_dirs"):
        cfg["lib_dirs"] = sc["lib_dirs"]
        print("[finalize] adopting lib_dirs from the setup-written config")


def apply_calibrated(args: list[str], settings: dict[str, str]) -> list[str]:
    out = list(args)
    for flag, value in settings.items():
        if flag in out:
            out[out.index(flag) + 1] = value
        else:
            out += [flag, value]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Resolve a Strata run-config template for this install.")
    ap.add_argument("template")
    ap.add_argument("--out", required=True)
    ap.add_argument("--strata-dir", default=str(REPO / "Strata"))
    ap.add_argument("--data-dir", default=str(REPO / "Strata-data"))
    ap.add_argument("--gguf-dir", default=None, help="folder holding both IQ3_S shards (default: <data-dir>/models)")
    ap.add_argument("--key-file", default=None, help="file holding the api_key value (default: <strata-dir>/api_key.txt)")
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--gpu-ids", default=None, help='override the config\'s "gpu" (nvidia-smi numbering, e.g. "1,0")')
    a = ap.parse_args()

    strata_dir = Path(a.strata_dir).resolve()
    data_dir = Path(a.data_dir).resolve()
    gguf_dir = Path(a.gguf_dir).resolve() if a.gguf_dir else data_dir / "models"
    key_file = Path(a.key_file) if a.key_file else strata_dir / "api_key.txt"

    cfg = load_json(Path(a.template))
    cfg.pop("_comment", None)
    cfg = resolve_tokens(cfg, {
        "STRATA_DIR": str(strata_dir),
        "DATA_DIR": str(data_dir),
        "GGUF_DIR": str(gguf_dir),
    })

    if a.gpu_ids:
        ids = [int(x) for x in a.gpu_ids.split(",")]
        cfg["gpu"] = ids[0] if len(ids) == 1 else ids
        # a single card cannot use the two-verifier pipeline of the split
        args = list(cfg.get("args", []))
        if len(ids) == 1 and "--pipeline-windows" in args:
            i = args.index("--pipeline-windows")
            del args[i:i + 2]
        cfg["args"] = args

    # api key: read from the local key file, inject, never echo
    key = key_file.read_text(encoding="utf-8").strip() if key_file.is_file() else ""
    cfg["api_key"] = key
    host = str(cfg.get("host", "127.0.0.1"))
    if host == "0.0.0.0" and len(key) < 32:
        print(f"error: host is 0.0.0.0 but no usable api_key in {key_file} — run ./install-strata.sh first "
              "(it generates the key locally) or set \"host\": \"127.0.0.1\" in the template", file=sys.stderr)
        return 1

    # per-PC calibration beats the author's numbers on a different box
    adopt_from_setup(cfg, strata_dir)

    if a.port:
        cfg["port"] = a.port

    blob = json.dumps(cfg)
    leftover = sorted(set(ANY_TOKEN_RE.findall(blob)))
    if leftover:
        print(f"error: unresolved tokens after finalize: {', '.join(leftover)} "
              "(check the template and --strata-dir/--data-dir/--gguf-dir)", file=sys.stderr)
        return 1
    if "<generated-at-install-time>" in blob:
        print("error: the api_key placeholder was not replaced", file=sys.stderr)
        return 1

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cfg, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    ctx = flag_value(list(cfg.get("args", [])), "--max-context")
    print(f"[finalize] wrote {out}  (port {cfg.get('port')}, {gpu_count(cfg)} GPU(s), context {ctx}, "
          f"api_key injected from {key_file.name})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
