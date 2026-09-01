"""Stage 18 — nhl `player_box` season dataset (parse + compile ONE family).

Thin numbered entry over ``nhl_data_build.season`` restricted to the ``player_box``
family (``--families player_box`` injected; remaining args forward verbatim).
Single home for the family list: the config DATASETS registry — these
stages are generated from it. Publish is the final numbered stage.

Usage::

    python -m nhl_data_18_player_box_creation -s 2026 --final-dir _raw/nhl/json/final
    scripts/nhl_data.sh 18
"""
from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    from nhl_data_build.season import main as _main

    argv = list(argv) if argv is not None else sys.argv[1:]
    return _main([*argv, "--families", "player_box"])


if __name__ == "__main__":
    raise SystemExit(main())
