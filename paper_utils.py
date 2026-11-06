"""Portable output paths and command-line parsing for the curated scripts."""

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path


def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {"true", "yes", "1"}:
        return True
    if value.lower() in {"false", "no", "0"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected true or false, got {value!r}")


def add_output_argument(parser):
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(__file__).resolve().parent / "artifacts",
        help="Root for generated results and figures (never committed).",
    )


def prepare_output(args, *, experiment=False):
    root = args.output_dir.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    (root / "graphs").mkdir(exist_ok=True)
    if experiment:
        config_dir = root / "configs"
        config_dir.mkdir(exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        (config_dir / f"{stamp}-{os.getpid()}.json").write_text(
            json.dumps(vars(args), indent=2, default=str) + "\n"
        )
        # The original experiment engines save arrays under relative directories.
        os.chdir(root)
    return root
