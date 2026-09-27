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


def r_squared(xs: list[float], ys: list[float]) -> float:
    """Share of y variance on the fitted line. Same number for all four series."""
    value = pearson(xs, ys)
    return value * value


def pearson(xs: list[float], ys: list[float]) -> float:
    mx, my = _mean(xs), _mean(ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    return sxy / (sxx * syy) ** 0.5


def _fit_line(points: list[tuple[float, float]]) -> tuple[float, float]:
    xs = [x for x, _ in points]
    ys = [y for _, y in points]
    mx, my = _mean(xs), _mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    return slope, my - slope * mx


def max_abs_residual(points: list[tuple[float, float]]) -> float:
    """Largest gap between a point and the least-squares line."""
    slope, intercept = _fit_line(points)
    return max(abs(y - (intercept + slope * x)) for x, y in points)


def points_beyond(points: list[tuple[float, float]], gap: float = 2.0) -> int:
    """How many points sit farther than `gap` from the fitted line."""
    slope, intercept = _fit_line(points)
    return sum(abs(y - (intercept + slope * x)) > gap for x, y in points)


def farthest_point(points: list[tuple[float, float]]) -> tuple[float, float]:
    """The point that sits farthest from the least-squares line."""
    slope, intercept = _fit_line(points)
    return max(points, key=lambda point: abs(point[1] - (intercept + slope * point[0])))


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
        "r2": r_squared(xs, ys),
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


def format_residuals(sets: dict[str, list[tuple[float, float]]]) -> str:
    lines = ["set   max_abs_residual   point"]
    for name in ORDER:
        x, y = farthest_point(sets[name])
        lines.append(f"{name:<4} {max_abs_residual(sets[name]):8.3f}   {x:.3f},{y:.3f}")
    return "\n".join(lines)


def scatter_svg(sets: dict[str, list[tuple[float, float]]]) -> str:
    """Four panels. Same axes, so the shared summary does not hide the shape."""
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="640" height="180" viewBox="0 0 640 180">',
        '<rect width="640" height="180" fill="#f7f4ef"/>',
    ]
    for index, name in enumerate(ORDER):
        left = 16 + index * 156
        parts.append(f'<text x="{left}" y="18" font-size="14" font-family="sans-serif">{name}</text>')
        far = farthest_point(sets[name])
        for x, y in sets[name]:
            px = left + (x - 3) / 17 * 130
            py = 160 - (y - 2) / 12 * 130
            if (x, y) == far:
                parts.append(
                    f'<circle cx="{px:.1f}" cy="{py:.1f}" r="5.2" fill="none" stroke="#c4552a" stroke-width="1.4"/>'
                )
            parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="3.2" fill="#1f4b4a"/>')
    parts.append("</svg>")
    return "\n".join(parts)


def main() -> None:
    sets = load_sets()
    print(format_report(sets))
    print(format_residuals(sets))
    (ROOT / "anscombe.svg").write_text(scatter_svg(sets), encoding="utf-8")


if __name__ == "__main__":
    main()
