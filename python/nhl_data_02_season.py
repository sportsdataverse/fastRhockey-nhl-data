"""Stage 02 — season dataset compile (parquet + rds + csv per family).

Thin numbered entry over ``nhl_data_build.season``; args forward verbatim. The library
package owns the logic; this file makes the repo lifecycle enumerable:
fetch -> season -> publish (models: nhl_model_NN_*). Single home:
models/manifest.yaml.

Usage::

    python -m nhl_data_02_season -s 2026 --final-dir _raw/nhl/json/final --out-dir nhl
    scripts/nhl_data.sh 02
"""
from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    from nhl_data_build.season import main as _main

    argv = list(argv) if argv is not None else sys.argv[1:]
    return _main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
