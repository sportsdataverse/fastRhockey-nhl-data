"""Shared runner for the numbered per-model stage scripts (``nhl_model_NN_*``, flat in python/).

Each stage is a thin, individually-runnable pipeline for ONE model:
fingerprint (skip unless ``--force``) -> train into a CANDIDATE dir -> append
``models/ledger.jsonl`` -> promote to the champion path only when the gate
passes -> record the fingerprint. Run via ``python -m nhl_model_NN_...`` or
``scripts/nhl_models.sh``.

**Candidate/promote.** The trainer used to write straight to ``models/`` and the
gate was evaluated afterwards, so a run that FAILED its floor still replaced the
committed champion (observed 2026-09-02: the stage-01 5v5 retrain scored
cv_auc 0.7786 with ``gate_pass: false`` and overwrote ``models/xg_model_5v5.json``
in place). Training now happens in ``models/.candidate/<stage>/``; a failing run
leaves every champion file byte-identical, still appends its ledger row, and
keeps the candidate on disk for inspection.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Callable, Optional

from nhl_data_build import fingerprint as fp

REPO_ROOT = Path(__file__).resolve().parents[2]
LEDGER = REPO_ROOT / "models" / "ledger.jsonl"
CANDIDATE_DIR = ".candidate"


def _seed_candidate(champion_dir: Path, candidate: Path) -> None:
    """Start the candidate from the current champion files (minus dotfiles).

    ``train_xg_models`` MERGES a single-variant run into the sidecars it finds in
    its output dir, so an empty candidate dir would drop the sibling variant's
    ``info_*`` / split entries. The fingerprint store (a dotfile) is deliberately
    not copied -- it belongs to the champion dir only.
    """
    if candidate.exists():
        shutil.rmtree(candidate)
    candidate.mkdir(parents=True, exist_ok=True)
    if not champion_dir.is_dir():
        return
    for src in champion_dir.iterdir():
        if src.is_file() and not src.name.startswith("."):
            shutil.copy2(src, candidate / src.name)


def run_stage(
    *,
    name: str,
    suite: str,
    config: dict,
    artifacts: list[Path],
    train: Callable[[Path], Optional[dict]],
    force: bool = False,
    smoke: bool = False,
) -> int:
    """Fingerprint-gated train run for one model; returns a process rc.

    ``train`` is called with the candidate output directory and must write its
    artifacts there (never to the champion dir). It returns ``None`` (no gates)
    or a dict carrying ``gate_pass``. Promotion happens only when ``gate_pass``
    is not ``False``; the fingerprint is recorded only after a promotion, so a
    failed run is retried rather than skipped as "already done".
    """
    suite_dir = REPO_ROOT / "python" / suite
    champion_dir = artifacts[0].parent
    store = champion_dir / fp.FINGERPRINT_STORE
    digest = fp.compute(suite_dir, config)
    if fp.should_skip(store, name, digest, artifacts, force):
        print(f"[{name}] fingerprint unchanged + artifacts present -> skip (--force to retrain)")
        return 0

    candidate = champion_dir / CANDIDATE_DIR / name
    _seed_candidate(champion_dir, candidate)
    gates = train(candidate)

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
        print(f"[{name}] NOT promoted -- champion files under {champion_dir} are unchanged.")
        print(f"[{name}] candidate kept for inspection: {candidate}")
        return 0 if smoke else 1

    missing = [a for a in artifacts if not (candidate / a.name).is_file()]
    if missing:
        print(f"[{name}] ERROR: expected artifact(s) not written: {[a.name for a in missing]}")
        return 1
    for src in sorted(candidate.iterdir()):
        if src.is_file() and not src.name.startswith("."):
            shutil.copy2(src, champion_dir / src.name)
    shutil.rmtree(candidate, ignore_errors=True)
    fp.record(store, name, digest)
    print(f"[{name}] done -> promoted {[str(a) for a in artifacts]}")
    return 0
