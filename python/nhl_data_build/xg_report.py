"""xG model training report - the summary table + feature importance from a training run.

Port of the summary block in ``build_xg_model.R`` (the CV/test logloss + AUC table, the
penalty-shot constant, era groupings, and the per-model feature-importance figures), driven
off the ``meta`` dict ``train_xg_models`` returns. Markdown is dependency-free; the PNG
figures are best-effort (need matplotlib).
"""

from __future__ import annotations

from pathlib import Path

_VARIANTS = [("5v5", "5v5"), ("st", "Special teams")]
_ERAS = [
    ("era_2011_2013", "2010-11 through 2012-13"),
    ("era_2014_2018", "2013-14 through 2017-18"),
    ("era_2019_2021", "2018-19 through 2020-21"),
    ("era_2022_2024", "2021-22 through 2023-24"),
    ("era_2025_on", "2024-25 and beyond"),
]


def _fmt(v: object) -> str:
    return f"{v:.4f}" if isinstance(v, float) else ("-" if v is None else str(v))


def build_report(meta: dict) -> str:
    """Render the training report markdown from a ``train_xg_models`` ``meta`` dict."""
    lines = [
        "# fastRhockey xG model training report",
        "",
        "Three models keyed off `strength_state`: a 5v5 model (all non-shootout/non-PS unblocked",
        "shots), a special-teams model, and a penalty-shot xG constant. Grouped 80/20 split + 5-fold",
        "grouped CV + min_child_weight grid (XGBoost `binary:logistic`).",
        "",
        "| Model | train / test | min_child_weight | nrounds | CV logloss | CV AUC | Test logloss | Test AUC |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key, label in _VARIANTS:
        i = meta.get(f"info_{key}")
        if not i:
            continue
        lines.append(
            f"| {label} | {i['n_train']} / {i['n_test']} | {i['min_child_weight']} | {i['nrounds']} | "
            f"{_fmt(i['cv_logloss'])} | {_fmt(i.get('cv_auc'))} | {_fmt(i['test_logloss'])} | {_fmt(i['test_auc'])} |"
        )
    lines.append(f"| Penalty shot | - | - | - | - | - | - | constant **{_fmt(meta.get('xg_model_ps'))}** |")

    lines += ["", "## Era groupings", ""]
    lines += [f"- `{name}` - {span}" for name, span in _ERAS]

    for key, label in _VARIANTS:
        i = meta.get(f"info_{key}")
        if not i or not i.get("importance"):
            continue
        lines += [
            "",
            f"## {label} feature importance (top {len(i['importance'])}, by gain)",
            "",
            f"Goal rate {_fmt(i.get('goal_rate'))} | {len(meta.get(f'xg_feature_names_{key}', []))} features.",
            "",
            "| feature | gain |",
            "|---|---:|",
        ]
        lines += [f"| `{row['feature']}` | {row['gain']} |" for row in i["importance"]]
    return "\n".join(lines) + "\n"


def write_xg_report(meta: dict, out_dir: str | Path, name: str = "xg_model_report.md") -> Path:
    path = Path(out_dir) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_report(meta), encoding="utf-8")
    return path


def write_importance_figures(meta: dict, fig_dir: str | Path) -> bool:
    """Per-model horizontal feature-importance bar charts (best-effort; needs matplotlib)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    for key, label in _VARIANTS:
        imp = (meta.get(f"info_{key}") or {}).get("importance")
        if not imp:
            continue
        rows = list(reversed(imp))
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh([r["feature"] for r in rows], [r["gain"] for r in rows], color="#99D9D9", edgecolor="#001628")
        ax.set_xlabel("Importance (gain)")
        ax.set_title(f"fastRhockey {label} xG model - feature importance")
        fig.tight_layout()
        fig.savefig(fig_dir / f"fastRhockey_xg_{key}_feature_importance.png", dpi=150)
        plt.close(fig)
    return True
