"""Stage 16 — nhl `shootout` season dataset (parse + compile ONE family).

Thin numbered entry over ``nhl_data_build.season`` restricted to the ``shootout``
family (``--families shootout`` injected; remaining args forward verbatim).
Single home for the family list: the config DATASETS registry — these
stages are generated from it. Publish is the final numbered stage.

Usage::

    python -m nhl_data_16_shootout_creation -s 2026 --final-dir _raw/nhl/json/final
    scripts/nhl_data.sh 16
"""
from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    from nhl_data_build.season import main as _main

    argv = list(argv) if argv is not None else sys.argv[1:]
    return _main([*argv, "--families", "shootout"])


if __name__ == "__main__":
    raise SystemExit(main())
