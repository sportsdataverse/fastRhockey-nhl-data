"""Publish the committed xG boosters to the sportsdataverse-data ``nhl_xg_models`` release.

The release is what fastRhockey and sportsdataverse-py score with. Its
``xg_model_meta.json`` is the MANIFEST: the penalty-shot constant, a
``version`` and the sha256 of every other asset. A consumer checks its cached
copy against the manifest on first use, so a retrain reaches installed copies
without a package release, and a cached file from another training (or a
partial download) is replaced instead of scored with. Before this the hashes
were pinned in fastRhockey's source, so every retrain broke installed copies.

Assets are built from ``models/`` (the promoted champion, committed by a
merged PR) -- never from a run artifact -- and the manifest is uploaded LAST, so
a consumer that reads the new manifest finds the files it names already there.

Usage::

    python -m nhl_data_build.xg_publish            # build into dist/nhl_xg_models, no upload
    python -m nhl_data_build.xg_publish --upload   # build, upload (--clobber), verify digests
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from nhl_data_build import publish

TAG = "nhl_xg_models"
REPO_ROOT = Path(__file__).resolve().parents[2]
VARIANTS = ("5v5", "st")
MANIFEST = "xg_model_meta.json"


def build(models: Path, out: Path, *, source: str) -> dict:
    """Write the release assets for the champion in *models* into *out*; returns the manifest."""
    meta = json.loads((models / MANIFEST).read_text(encoding="utf-8"))
    out.mkdir(parents=True, exist_ok=True)
    for v in VARIANTS:
        booster = (models / f"xg_model_{v}.json").read_bytes()
        names = json.loads(booster)["learner"]["feature_names"]
        if names != meta[f"xg_feature_names_{v}"]:
            raise ValueError(f"{v}: booster feature names differ from {MANIFEST} -- the sidecar is from another fit")
        (out / f"xg_model_{v}.json").write_bytes(booster)
        # one name per line, booster order: the R scorer selects columns in this order
        (out / f"feats_{v}.txt").write_bytes(("\n".join(names) + "\n").encode("utf-8"))
    files = {p.name: publish._sha256(p).removeprefix("sha256:") for p in sorted(out.iterdir()) if p.name != MANIFEST}
    trained = {v: meta[f"info_{v}"]["trained_at"] for v in VARIANTS}
    manifest = {
        "xg_model_ps": meta["xg_model_ps"],
        # ponytail: the latest training date; two promotions on one day share a version
        "version": max(trained.values())[:10],
        "trained_at": trained,
        "source": source,
        "files": files,
    }
    (out / MANIFEST).write_bytes((json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    return manifest


def upload(out: Path, manifest: dict, *, repo: str = publish._REPO) -> None:
    """Upload the assets (manifest last), stamp the timestamp sidecars, then check the digests."""
    for name in sorted(manifest["files"]):
        publish.publish_file(out / name, TAG, repo=repo)
    publish.publish_file(out / MANIFEST, TAG, repo=repo)
    publish.upload_release_sidecars(TAG, runner=publish._gh, repo=repo)
    live = publish._release_digests(TAG, repo) or {}
    want = {**{n: f"sha256:{h}" for n, h in manifest["files"].items()}, MANIFEST: publish._sha256(out / MANIFEST)}
    bad = {n: live.get(n) for n, h in want.items() if live.get(n) != h}
    if bad:
        raise RuntimeError(f"release digests do not match what was uploaded: {bad}")


def _source() -> str:
    """``<repo>@<commit>`` of the committed champion; refuses uncommitted models."""
    git = ["git", "-C", str(REPO_ROOT)]
    if subprocess.run(
        [*git, "status", "--porcelain", "--", "models"], capture_output=True, text=True, check=True
    ).stdout.strip():
        raise SystemExit("models/ has uncommitted changes -- publish only a merged champion")
    sha = subprocess.run([*git, "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    return f"sportsdataverse/fastRhockey-nhl-data@{sha}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m nhl_data_build.xg_publish")
    ap.add_argument("--models", default=str(REPO_ROOT / "models"))
    ap.add_argument("--out", default=str(REPO_ROOT / "dist" / TAG))
    ap.add_argument("--upload", action="store_true", help="upload to the release (--clobber) and verify digests")
    args = ap.parse_args(argv)
    out = Path(args.out)
    manifest = build(Path(args.models), out, source=_source())
    print(json.dumps(manifest, indent=2, sort_keys=True))
    if args.upload:
        upload(out, manifest)
        print(f"uploaded {len(manifest['files']) + 1} assets to {TAG}; digests verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
