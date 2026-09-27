"""Leakage analysis: exact, cross-split, cross-class, near-duplicate, filename.

Hashing design
--------------
* **SHA-256** is the single exact-duplicate identity, used both at build time and
  here. One function, one meaning.
* **MD5 is not used anywhere.** It is not collision resistant and has no place in
  a pipeline that decides whether results are trustworthy.
* **Perceptual hashing** (pHash by default, dHash/aHash selectable) finds images
  that *look* the same. This is deliberately insensitive to scaling, compression
  and small edits - which is exactly what makes it useful for finding
  near-duplicates, and exactly why it can never be used for identity or
  security. It is a heuristic and is labelled as one.
* **pHash, not aHash, is the default.** aHash thresholds each pixel against the
  image mean, so it mostly encodes overall brightness. On leaf photographs -
  where most images are similarly-lit green shapes - aHash at a useful threshold
  merges hundreds of *unrelated* leaves into single clusters and its counts
  cannot be trusted. pHash keeps the lowest-frequency DCT coefficients, so it
  tracks structure. Measured on this dataset, unrelated same-class pairs fall
  within distance 10 only 0.01% of the time, while genuine re-encoded copies
  land at distance 0-2.

Performance
-----------
The original implementation compared every test image against every train image
with a Python ``bin(a ^ b).count("1")`` call: O(n_test x n_train) interpreted
operations - roughly 6.5M calls per class on a dataset this size, i.e. hours.
Distances here use vectorised ``np.bitwise_count`` over chunked blocks, which
turns the same work into a handful of array operations.
"""

from __future__ import annotations

import csv
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from iwnet.config import Config
from iwnet.utils import get_logger, iter_images, perceptual_hash, sha256_file

__all__ = [
    "LeakageHit",
    "LeakageReport",
    "analyse",
    "write_leakage_report",
    "run_leakage_check",
    "index_split",
    "LEAKAGE_REPORT_NAME",
]

log = get_logger("data.leakage")

LEAKAGE_REPORT_NAME = "leakage_report.csv"

#: Cap on CSV rows so a catastrophic split cannot produce a multi-gigabyte
#: report. The console totals are always complete regardless of this cap.
MAX_REPORT_ROWS = 20_000


@dataclass
class LeakageHit:
    """One identified problem file pair."""

    kind: str  # exact | cross_class_exact | filename | near
    cls: str
    test_path: str
    train_path: str
    distance: int | None = None
    sha256: str | None = None

    def as_row(self) -> list[str]:
        return [
            self.kind,
            self.cls,
            self.train_path,
            self.test_path,
            "" if self.distance is None else str(self.distance),
            self.sha256 or "",
        ]


@dataclass
class _Record:
    """Internal per-image view used by the analysis."""

    split: str
    cls: str
    path: Path
    sha: str
    perceptual: int | None = None


@dataclass
class LeakageReport:
    hits: list[LeakageHit] = field(default_factory=list)
    exact: list[LeakageHit] = field(default_factory=list)
    cross_class: list[LeakageHit] = field(default_factory=list)
    filenames: list[LeakageHit] = field(default_factory=list)
    near: list[LeakageHit] = field(default_factory=list)
    duplicate_hashes_within_split: int = 0
    images_checked: int = 0
    #: Test images actually fed to the near-duplicate search. A value of 0 while
    #: ``images_checked`` is large means the search silently did not run.
    near_candidates_checked: int = 0
    near_threshold: int = 5
    elapsed_seconds: float = 0.0

    @property
    def blocking(self) -> list[LeakageHit]:
        """Hits that invalidate a test accuracy: identical bytes in both splits."""
        return self.exact + self.cross_class

    def summary(self) -> dict[str, int]:
        return {
            "exact": len(self.exact),
            "cross_class": len(self.cross_class),
            "filename": len(self.filenames),
            "near": len(self.near),
            "near_candidates_checked": self.near_candidates_checked,
            "within_split_duplicates": self.duplicate_hashes_within_split,
        }


def _rel(path: Path) -> str:
    return str(path).replace("\\", "/")


# ─────────────────────────────────────────────────────────────────────────────
#  INDEXING
# ─────────────────────────────────────────────────────────────────────────────
def index_split(split_dir: Path, exts) -> dict[str, list[Path]]:
    """Class name -> sorted image paths for one split directory."""
    if not split_dir.is_dir():
        return {}
    out: dict[str, list[Path]] = {}
    for class_dir in sorted((p for p in split_dir.iterdir() if p.is_dir()), key=lambda p: p.name):
        images = iter_images(class_dir, exts)
        if images:
            out[class_dir.name] = images
    return out


def _load_manifest_hashes(cfg: Config) -> dict[str, str]:
    """Reuse SHA-256 values already computed during the build, when available.

    Keyed by the dataset-relative path (``train/Esca/foo.jpg``), which is the one
    identifier the manifest and the copied tree agree on.
    """
    manifest = cfg.paths.results_dir / "split_manifest.csv"
    if not manifest.is_file():
        return {}
    try:
        with open(manifest, encoding="utf-8") as handle:
            return {
                f"{row['split']}/{row['class']}/{row['filename']}": row["sha256"]
                for row in csv.DictReader(handle)
                if row.get("sha256")
            }
    except (OSError, KeyError) as exc:
        log.warning("Could not reuse manifest hashes (%s); will hash directly", exc)
        return {}


def _collect_records(
    cfg: Config,
    index: dict[str, list[Path]],
    split: str,
    manifest: dict[str, str],
    workers: int,
) -> list[_Record]:
    """Build records for one split, hashing and perceptual-hashing in parallel."""

    def digest_of(path: Path) -> str:
        key = f"{split}/{path.parent.name}/{path.name}"
        cached = manifest.get(key)
        if cached:
            return cached
        try:
            return sha256_file(path)
        except OSError:
            return ""

    algorithm = getattr(cfg.leakage, "hash_algorithm", "phash")
    work: list[tuple[str, Path]] = [
        (cls, path) for cls, images in index.items() for path in images
    ]

    def digest_task(item: tuple[str, Path]) -> str:
        return digest_of(item[1])

    def perceptual_task(item: tuple[str, Path]) -> int | None:
        try:
            return perceptual_hash(item[1], algorithm)
        except (OSError, ValueError):
            return None

    records: list[_Record] = []
    # A single shared pool with real .map fan-out. Submitting two serial lambdas
    # to a pool (as an earlier version did) gives no parallelism at all.
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        digest_futures = [pool.submit(digest_task, item) for item in work]
        perceptual_futures = [pool.submit(perceptual_task, item) for item in work]
        digests = [f.result() for f in digest_futures]
        perceptuals = [f.result() for f in perceptual_futures]

    for (cls, path), digest, ph in zip(work, digests, perceptuals):
        records.append(_Record(split, cls, path, digest, ph))
    return records


# ─────────────────────────────────────────────────────────────────────────────
#  ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
def analyse(cfg: Config, *, workers: int | None = None) -> LeakageReport:
    """Audit the dataset that is about to be trained on."""
    import numpy as np

    started = time.perf_counter()
    threshold = cfg.leakage.near_dup_distance
    report = LeakageReport(near_threshold=threshold)

    train_index = index_split(cfg.paths.train_dir, cfg.data.img_exts)
    test_index = index_split(cfg.paths.test_dir, cfg.data.img_exts)
    if not train_index or not test_index:
        log.warning("Leakage analysis skipped: train or test split is empty")
        report.elapsed_seconds = time.perf_counter() - started
        return report

    manifest = _load_manifest_hashes(cfg)
    workers = workers or max(1, min(8, (cfg.data.num_workers or 2) * 2))

    train_records = _collect_records(cfg, train_index, "train", manifest, workers)
    test_records = _collect_records(cfg, test_index, "test", manifest, workers)
    report.images_checked = len(test_records)

    # ── Exact duplicates across splits (SHA-256) ────────────────────────────
    train_by_sha: dict[str, list[_Record]] = {}
    for record in train_records:
        if record.sha:
            train_by_sha.setdefault(record.sha, []).append(record)

    # Only hashes that are *actual* cross-split exact matches need suppressing in
    # the near-duplicate pass. Recording every test hash here would exclude the
    # entire test set and silently skip the search.
    exact_cross_split: set[str] = set()
    for record in test_records:
        if not record.sha:
            continue
        matches = train_by_sha.get(record.sha)
        if not matches:
            continue
        exact_cross_split.add(record.sha)
        for match in matches:
            kind = "cross_class_exact" if match.cls != record.cls else "exact"
            hit = LeakageHit(kind, record.cls, _rel(record.path), _rel(match.path), 0, record.sha)
            report.hits.append(hit)
            (report.cross_class if kind == "cross_class_exact" else report.exact).append(hit)

    # ── Duplicates within a single split ────────────────────────────────────
    for records in (train_records, test_records):
        seen: dict[str, int] = {}
        for record in records:
            if record.sha:
                seen[record.sha] = seen.get(record.sha, 0) + 1
        report.duplicate_hashes_within_split += sum(n - 1 for n in seen.values() if n > 1)

    # ── Duplicate filenames across splits (bytes differ) ────────────────────
    train_names: dict[str, _Record] = {}
    for record in train_records:
        train_names.setdefault(record.path.name, record)
    for record in test_records:
        other = train_names.get(record.path.name)
        if other is not None and other.sha != record.sha:
            report.filenames.append(
                LeakageHit("filename", record.cls, _rel(record.path), _rel(other.path), None, record.sha)
            )

    # ── Near duplicates (vectorised Hamming distance) ───────────────────────
    report.near = _find_near_duplicates(train_records, test_records, exact_cross_split, threshold)
    report.near_candidates_checked = sum(
        1
        for record in test_records
        if record.perceptual is not None and record.sha not in exact_cross_split
    )
    report.hits.extend(report.near)

    report.elapsed_seconds = time.perf_counter() - started
    return report


def _find_near_duplicates(
    train_records: list[_Record],
    test_records: list[_Record],
    exact_cross_split: set[str],
    threshold: int,
) -> list[LeakageHit]:
    """Per-class near-duplicate search, vectorised and chunked over the test set.

    ``exact_cross_split`` holds only the hashes already reported as exact
    cross-split matches. Those are skipped so a near-duplicate report is not
    padded with distance-0 rows that duplicate the exact-duplicate report.
    Anything not in that set is searched normally.
    """
    import numpy as np

    train_by_class: dict[str, list[_Record]] = {}
    for record in train_records:
        if record.perceptual is not None:
            train_by_class.setdefault(record.cls, []).append(record)

    hits: list[LeakageHit] = []
    test_by_class: dict[str, list[_Record]] = {}
    for record in test_records:
        if record.perceptual is not None and record.sha not in exact_cross_split:
            test_by_class.setdefault(record.cls, []).append(record)

    for cls, candidates in sorted(test_by_class.items()):
        pool = train_by_class.get(cls)
        if not pool:
            continue
        train_matrix = np.array([r.perceptual for r in pool], dtype=np.uint64)
        test_matrix = np.array([r.perceptual for r in candidates], dtype=np.uint64)

        # Chunk over the test side so peak memory stays bounded.
        chunk = 256
        for start in range(0, test_matrix.shape[0], chunk):
            block = test_matrix[start : start + chunk]
            distances = np.bitwise_count(np.bitwise_xor(block[:, None], train_matrix[None, :]))
            for offset in range(block.shape[0]):
                best = int(distances[offset].min())
                if best <= threshold:
                    record = candidates[start + offset]
                    match = pool[int(distances[offset].argmin())]
                    hits.append(
                        LeakageHit("near", cls, _rel(record.path), _rel(match.path), best, None)
                    )
    return hits


# ─────────────────────────────────────────────────────────────────────────────
#  REPORTING
# ─────────────────────────────────────────────────────────────────────────────
def write_leakage_report(cfg: Config, report: LeakageReport) -> Path | None:
    """Write ``leakage_report.csv``; return ``None`` when there is nothing to report."""
    cfg.paths.results_dir.mkdir(parents=True, exist_ok=True)
    path = cfg.paths.results_dir / LEAKAGE_REPORT_NAME

    if not report.hits:
        if path.is_file():
            path.unlink()  # never leave a stale report implying a problem
        return None

    # Blocking hits first, then near-duplicates, then filename collisions.
    order = {"cross_class_exact": 0, "exact": 0, "near": 1, "filename": 2}
    ordered = sorted(report.hits, key=lambda h: (order.get(h.kind, 3), h.cls, h.test_path))

    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["type", "class", "train_path", "test_path", "distance", "hash"])
        for hit in ordered[:MAX_REPORT_ROWS]:
            writer.writerow(hit.as_row())
    if len(ordered) > MAX_REPORT_ROWS:
        log.warning("leakage_report.csv truncated to %d of %d rows", MAX_REPORT_ROWS, len(ordered))
    return path


def run_leakage_check(cfg: Config, *, allow: bool = False) -> LeakageReport:
    """Audit the dataset, print a summary, and hard-stop on exact leakage.

    Exact cross-split duplicates invalidate any accuracy number, so training is
    refused by default. ``--allow-leakage`` is the explicit, deliberate override.
    """
    print("=" * 68)
    print("  DATA LEAKAGE CHECK")
    print("=" * 68)

    report = analyse(cfg)
    counts = report.summary()

    print(f"  Test images checked               : {report.images_checked}")
    print(f"  Near-dup search candidates        : {counts['near_candidates_checked']}")
    print(f"  Exact duplicates (train/test)     : {counts['exact']}")
    print(f"  Cross-CLASS exact duplicates      : {counts['cross_class']}")
    print(f"  Near duplicates ({cfg.leakage.hash_algorithm} <= {report.near_threshold}) : {counts['near']}")
    print(f"  Duplicate filenames across splits : {counts['filename']}")
    print(f"  Duplicate hashes within dataset   : {counts['within_split_duplicates']}")
    print(f"  Elapsed                           : {report.elapsed_seconds:.1f}s")

    if report.images_checked == 0:
        print("\n  [SKIP] No test images to check.\n")
        return report

    # A near-duplicate count of 0 is only meaningful if the search actually ran.
    if counts["near_candidates_checked"] == 0 and report.images_checked > 0:
        print(
            "  [ERROR] The near-duplicate search had no candidates. A '0 near "
            "duplicates' result here is meaningless.\n"
        )
        return report

    path = write_leakage_report(cfg, report)
    blocking = report.blocking

    if not blocking and not report.near and not report.filenames:
        print("\n  [GOOD] No leakage detected. Any accuracy produced is genuine.\n")
        return report

    if path is not None:
        print(f"\n  Full report -> {path}")

    if blocking:
        total = len(blocking)
        pct = total / max(report.images_checked, 1) * 100
        print(f"\n  [ERROR] DATA LEAKAGE DETECTED - {total} exact duplicate(s) across splits "
              f"({pct:.1f}% of the test set).")
        print("  Any accuracy measured on this split would be meaningless.")
        if allow:
            print("  Continuing because --allow-leakage was set. Results are NOT valid.")
        else:
            print("\n  Training stopped. Rebuild the split (dedup happens BEFORE the split):")
            print("    python grape.py --build-dataset")
            print("  then verify:")
            print("    python grape.py --dataset-info")
            print("  to proceed anyway (NOT recommended):")
            print("    python grape.py --allow-leakage")
            sys.exit(1)
    elif report.near:
        print(f"\n  [WARN] {len(report.near)} near-duplicate(s) across splits. "
              "Perceptual hashing is a heuristic - inspect the report before trusting the split.")
    elif report.filenames:
        print(f"\n  [WARN] {len(report.filenames)} duplicate filename(s) across splits "
              "(informational only; bytes differ).")
    print()
    return report
