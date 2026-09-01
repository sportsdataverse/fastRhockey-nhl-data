"""Stage 01 — raw per-game finals fetch over HTTP (read-through cache, fail-soft per game).

Thin numbered entry over ``nhl_data_build.fetch``; args forward verbatim. The library
package owns the logic; this file makes the repo lifecycle enumerable:
fetch -> season -> publish (models: nhl_model_NN_*). Single home:
models/manifest.yaml.

Usage::

    python -m nhl_data_01_fetch -s 2026 --dest _raw/nhl/json/final
    scripts/nhl_data.sh 01
"""
from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    from nhl_data_build.fetch import main as _main

    argv = list(argv) if argv is not None else sys.argv[1:]
    return _main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
