"""Summarize Anscombe's four series. Numbers come from data/anscombe.csv."""

from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "anscombe.csv"
ORDER = ("I", "II", "III", "IV")


def load_sets(path: Path = DATA) -> dict[str, list[tuple[float, float]]]:
    sets: dict[str, list[tuple[float, float]]] = {name: [] for name in ORDER}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            name = row["set"].strip()
            if name not in sets:
                raise ValueError(f"unknown set {name}")
            sets[name].append((float(row["x"]), float(row["y"])))
    return sets


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def sample_variance(values: list[float]) -> float:
    center = _mean(values)
    return sum((value - center) ** 2 for value in values) / (len(values) - 1)


def pearson(xs: list[float], ys: list[float]) -> float:
    mx, my = _mean(xs), _mean(ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    return sxy / (sxx * syy) ** 0.5


def summarize(points: list[tuple[float, float]]) -> dict[str, float]:
    xs = [x for x, _ in points]
    ys = [y for _, y in points]
    return {
        "n": float(len(points)),
        "mean_x": _mean(xs),
        "mean_y": _mean(ys),
        "var_x": sample_variance(xs),
        "var_y": sample_variance(ys),
        "corr": pearson(xs, ys),
    }


def format_report(sets: dict[str, list[tuple[float, float]]]) -> str:
    lines = ["set   n   mean_x  mean_y   var_x   var_y   corr"]
    for name in ORDER:
        row = summarize(sets[name])
        lines.append(
            f"{name:<4} {int(row['n']):>2}  "
            f"{row['mean_x']:7.3f} {row['mean_y']:7.3f}  "
            f"{row['var_x']:6.3f}  {row['var_y']:6.3f}  {row['corr']:6.3f}"
        )
    return "\n".join(lines)


def main() -> None:
    print(format_report(load_sets()))


if __name__ == "__main__":
    main()
