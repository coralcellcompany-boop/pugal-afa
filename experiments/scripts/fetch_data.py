"""Fetch the raw real-data files into experiments/data/ and record SHA-256 hashes.

    uv run python scripts/fetch_data.py [--force] [--afabench-dir PATH]

- diabetes.csv, miniboone.csv, physionet.csv: copied from a local AFABench clone
  (extra/data/misc/) when present, else downloaded from the AFABench GitHub repo.
- bank.csv (UCI 222, bank-full), actg175.csv (UCI 890), ckd.csv (UCI 336): the
  `data.csv` export of the UCI ML Repository (the file `ucimlrepo` reads, which is
  what AFABench uses). Stored unchanged; preprocessing lives in src/experiment/data.py.

Downloaded files are treated as data only: they are parsed as CSV, never executed.
Writes data/SHA256SUMS (format of `shasum -a 256`; check with `shasum -a 256 -c SHA256SUMS`).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DEFAULT_CLONE = Path(
    "/private/tmp/claude-501/-Volumes-development-Projects-Researches/"
    "2accc55e-656c-4224-bc43-deead2a8db80/scratchpad/ext/AFA-Benchmark"
)
AFABENCH_RAW = "https://raw.githubusercontent.com/Linusaronsson/AFA-Benchmark/main/extra/data/misc/"
UCI = "https://archive.ics.uci.edu/static/public/{id}/data.csv"

# file name -> (source, expected last (label) column, expected number of columns)
AFABENCH_FILES = {
    "diabetes.csv": ("Outcome", 46),
    "miniboone.csv": ("Outcome", 51),
    "physionet.csv": ("Outcome", 42),
}
# file name -> (UCI id, label column, expected number of columns)
UCI_FILES = {
    "bank.csv": (222, "y", 17),
    "actg175.csv": (890, "cid", 25),
    "ckd.csv": (336, "class", 25),
}


def _download(url: str, dest: Path) -> None:
    print(f"  downloading {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "pugal-afa-fetch/1.0"})
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(req, timeout=300) as resp, open(tmp, "wb") as out:
        shutil.copyfileobj(resp, out)
    tmp.replace(dest)


def _header(path: Path) -> list[str]:
    with open(path, newline="", encoding="utf-8") as f:
        return next(csv.reader(f))


def _check(path: Path, label: str, n_cols: int, last_is_label: bool) -> None:
    head = _header(path)
    if len(head) != n_cols:
        raise SystemExit(f"{path.name}: expected {n_cols} columns, found {len(head)}")
    if last_is_label and head[-1] != label:
        raise SystemExit(f"{path.name}: last column is {head[-1]!r}, expected label {label!r}")
    if not last_is_label and label not in head:
        raise SystemExit(f"{path.name}: label column {label!r} missing")
    where = "last column" if last_is_label else f"column {head.index(label)}"
    print(f"  ok {path.name}: {n_cols} columns, label {label!r} ({where})")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="re-fetch files that already exist")
    ap.add_argument("--afabench-dir", type=Path, default=DEFAULT_CLONE, help="local AFABench clone")
    args = ap.parse_args()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    local = args.afabench_dir / "extra" / "data" / "misc"

    for fname, (label, n_cols) in AFABENCH_FILES.items():
        dest = DATA_DIR / fname
        if args.force or not dest.exists():
            if (local / fname).is_file():
                print(f"  copying {local / fname}")
                shutil.copyfile(local / fname, dest)
            else:
                _download(AFABENCH_RAW + fname, dest)
        _check(dest, label, n_cols, last_is_label=True)

    for fname, (uci_id, label, n_cols) in UCI_FILES.items():
        dest = DATA_DIR / fname
        if args.force or not dest.exists():
            _download(UCI.format(id=uci_id), dest)
        _check(dest, label, n_cols, last_is_label=False)

    files = sorted(p for p in DATA_DIR.iterdir() if p.is_file() and p.suffix == ".csv")
    lines = [f"{_sha256(p)}  {p.name}" for p in files]
    (DATA_DIR / "SHA256SUMS").write_text("\n".join(lines) + "\n")
    print(f"wrote {DATA_DIR / 'SHA256SUMS'}:")
    print("\n".join("  " + line for line in lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
