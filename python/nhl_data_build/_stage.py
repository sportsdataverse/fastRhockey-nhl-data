"""Shared runner for the numbered per-model stage scripts (``nhl_model_NN_*``, flat in python/).

Each stage is a thin, individually-runnable pipeline for ONE model:
fingerprint (skip unless ``--force``) -> train via ``nhl_data_build.xg_train``
-> record fingerprint -> append ``models/ledger.jsonl`` -> fail on a hard gate
miss. Run via ``python -m nhl_model_NN_...`` or ``scripts/nhl_models.sh``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from nhl_data_build import fingerprint as fp

REPO_ROOT = Path(__file__).resolve().parents[2]
LEDGER = REPO_ROOT / "models" / "ledger.jsonl"


def run_stage(
    *,
    name: str,
    suite: str,
    config: dict,
    artifacts: list[Path],
    train: Callable[[], Optional[dict]],
    force: bool = False,
    smoke: bool = False,
) -> int:
    """Fingerprint-gated train run for one model; returns a process rc."""
    suite_dir = REPO_ROOT / "python" / suite
    store = artifacts[0].parent / fp.FINGERPRINT_STORE
    digest = fp.compute(suite_dir, config)
    if fp.should_skip(store, name, digest, artifacts, force):
        print(f"[{name}] fingerprint unchanged + artifacts present -> skip (--force to retrain)")
        return 0

    gates = train()  # None, or a dict carrying gate_pass

    fp.record(store, name, digest)
    fp.append_ledger(
        LEDGER,
        {
            "suite": suite,
            "model": name,
            "fingerprint": digest,
            "config": config,
            "artifacts": [a.name for a in artifacts],
            "gates": gates,
            "delta_vs_champion": None,
            "in_published_data": False,  # flips when the next season compile ships the scores
        },
    )

    if gates is not None and gates.get("gate_pass") is False:
        print(f"[{name}] gate: FAIL" + (" (smoke run, tolerated)" if smoke else ""))
        if not smoke:
            return 1
    missing = [a for a in artifacts if not a.is_file()]
    if missing:
        print(f"[{name}] ERROR: expected artifact(s) not written: {missing}")
        return 1
    print(f"[{name}] done -> {[str(a) for a in artifacts]}")
    return 0
