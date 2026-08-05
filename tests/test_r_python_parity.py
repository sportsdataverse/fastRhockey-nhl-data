"""R <-> Python dataset-registry parity.

Standing policy (2026-08-03): this repo carries BOTH pipelines. Python is the
primary one and gets the work; the R chain is maintained as the methodological
/ language equivalent; **both move together when either changes.**

`python/nhl_data_build/config.py` says so itself -- "Dataset registry - Python
port of ``nhl_data_creation.R``'s ``DATASETS`` tribble" -- and that claim is
exactly what this module turns into a test. The two declarations carry the same
five fields per dataset (key, json_field, file_prefix, release_tag,
description), so drift is mechanically detectable without running either
pipeline: no R runtime, no network, no fixtures.

**Neither side is authoritative.** A failure here does not mean "fix R to match
Python" or the reverse -- it means the two pipelines disagree about what they
produce, and a human decides which is right. The assertion messages are written
to support that decision, not to pre-empt it.

What this does NOT prove: that the two pipelines produce the same *values* for
a dataset. That is the output-parity harness (run both over a fixture season
and compare frames), which is a separate, heavier phase. This is the cheap
contract-level guard that catches the common failure -- someone adds or renames
a dataset on one side only.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
R_SOURCE = REPO / "R" / "nhl_data_creation.R"
PY_SOURCE = REPO / "python" / "nhl_data_build" / "config.py"

FIELDS = ("key", "json_field", "file_prefix", "release_tag", "description")


def _python_datasets() -> list[tuple[str, ...]]:
    """The DATASETS list from config.py, read via AST (never imported).

    Importing would drag in the package's dependencies for what is a pure
    literal, and would let a side effect in config.py influence the test.
    """
    tree = ast.parse(PY_SOURCE.read_text(encoding="utf-8"))
    for node in tree.body:
        targets = [node.target] if isinstance(node, ast.AnnAssign) else getattr(node, "targets", [])
        if not any(isinstance(t, ast.Name) and t.id == "DATASETS" for t in targets):
            continue
        value = node.value
        assert isinstance(value, ast.List), "DATASETS is expected to be a list literal"
        rows = []
        for elt in value.elts:
            assert isinstance(elt, ast.Tuple), "each DATASETS row is expected to be a tuple"
            rows.append(tuple(ast.literal_eval(e) for e in elt.elts))
        return rows
    raise AssertionError(f"no DATASETS assignment found in {PY_SOURCE}")


# A tribble row is `"a", "b", ..., "e",` -- five double-quoted cells. Commas
# inside a cell would break this, so the parser asserts the cell count per row
# rather than trusting the split.
_TRIBBLE = re.compile(r"DATASETS\s*<-\s*tibble::tribble\((.*?)\n\)", re.S)
_CELL = re.compile(r'"((?:[^"\\]|\\.)*)"')


def _r_datasets() -> list[tuple[str, ...]]:
    """The DATASETS tribble from nhl_data_creation.R."""
    text = R_SOURCE.read_text(encoding="utf-8")
    match = _TRIBBLE.search(text)
    assert match, f"no `DATASETS <- tibble::tribble(` block found in {R_SOURCE}"

    rows = []
    for line in match.group(1).splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("~"):
            continue  # blank, comment, or the header row of ~names
        cells = _CELL.findall(stripped)
        if not cells:
            continue
        assert len(cells) == len(FIELDS), f"tribble row has {len(cells)} cells, expected {len(FIELDS)}: {stripped!r}"
        rows.append(tuple(cells))
    return rows


def _fmt(rows: list[tuple[str, ...]]) -> str:
    return "\n".join("  " + " | ".join(r) for r in rows)


def test_both_registries_parse():
    """Guard the guard: a parser that silently finds nothing would pass every
    comparison below vacuously."""
    assert len(_python_datasets()) > 0
    assert len(_r_datasets()) > 0


def test_same_datasets_are_declared():
    """Neither pipeline may gain or lose a dataset alone."""
    py = {r[0] for r in _python_datasets()}
    r = {r[0] for r in _r_datasets()}
    only_py = sorted(py - r)
    only_r = sorted(r - py)
    assert not (only_py or only_r), (
        "R and Python disagree about which datasets this repo produces.\n"
        f"  Python only: {only_py}\n"
        f"  R only:      {only_r}\n"
        "Neither side is automatically right -- decide which pipeline is correct, "
        "then update the other. Adding the dataset to both is usually the answer; "
        "deleting it from both is a release-contract change."
    )


@pytest.mark.parametrize("field_index,field", list(enumerate(FIELDS))[1:])
def test_each_field_agrees(field_index: int, field: str):
    """Per-field comparison so a failure names the column, not just 'they differ'.

    file_prefix and release_tag are the load-bearing ones -- they decide the
    filename and the GitHub release asset a consumer downloads, so a silent
    divergence there ships data to the wrong place.
    """
    py = {r[0]: r for r in _python_datasets()}
    r = {r[0]: r for r in _r_datasets()}
    mismatches = [
        (key, py[key][field_index], r[key][field_index])
        for key in sorted(set(py) & set(r))
        if py[key][field_index] != r[key][field_index]
    ]
    assert not mismatches, (
        f"R and Python disagree on `{field}` for {len(mismatches)} dataset(s):\n"
        + "\n".join(f"  {k}: python={p!r}  r={rr!r}" for k, p, rr in mismatches)
        + "\nNeither side is authoritative -- this is a review item."
    )


def test_row_order_matches():
    """Order is not merely cosmetic: both pipelines iterate this table, so the
    order decides build and upload sequence. Divergence means the two runs do
    the same work in different orders, which makes their logs and any
    partial-failure state hard to compare."""
    py = [r[0] for r in _python_datasets()]
    r = [r[0] for r in _r_datasets()]
    assert py == r, f"Dataset order differs between the two registries.\n  python: {py}\n  r:      {r}"
