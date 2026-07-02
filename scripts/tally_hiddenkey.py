"""Print mean ± std test_auc for every `hk-*` experiment in results.tsv."""

from __future__ import annotations

import statistics
from collections import defaultdict
from pathlib import Path

RESULTS = Path("results.tsv")
PREFIXES = ("hk-", "hkx-", "hky-")


def main() -> None:
    by_exp: dict[str, list[float]] = defaultdict(list)
    if not RESULTS.exists():
        print("results.tsv not found")
        return
    with RESULTS.open() as f:
        header = next(f, "").rstrip("\n").split("\t")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 7:
                continue
            exp = parts[1]
            if not exp.startswith(PREFIXES):
                continue
            try:
                by_exp[exp].append(float(parts[6]))
            except ValueError:
                continue

    if not by_exp:
        print(f"No rows matching {PREFIXES!r} yet.")
        return

    width = max(len(k) for k in by_exp)
    print(f"{'method':<{width}}  {'n':>3}  test_auc")
    for name in sorted(by_exp, key=lambda k: -statistics.mean(by_exp[k])):
        aucs = by_exp[name]
        mean = statistics.mean(aucs)
        std = statistics.pstdev(aucs) if len(aucs) > 1 else 0.0
        print(f"{name:<{width}}  {len(aucs):>3}  {mean:.4f} ± {std:.4f}")


if __name__ == "__main__":
    main()
