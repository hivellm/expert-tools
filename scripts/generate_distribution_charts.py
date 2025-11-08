#!/usr/bin/env python3
"""
Generate dataset distribution charts for expert-tools.

Creates source-distribution visualizations (bar + pie) using
`datasets/source_counts.json` as produced by preprocess.py.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt  # noqa: E402


SOURCE_FILE = Path("datasets/source_counts.json")
OUTPUT_DIR = Path("docs")


def load_counts() -> OrderedDict[str, int]:
    if not SOURCE_FILE.exists():
        raise FileNotFoundError(
            f"{SOURCE_FILE} not found. Run preprocess.py to generate counts."
        )
    counts = json.loads(SOURCE_FILE.read_text(encoding="utf-8"))
    ordered = OrderedDict(
        sorted(counts.items(), key=lambda item: item[1], reverse=True)
    )
    return ordered


def format_label(name: str) -> str:
    return name.replace("interstellarninja/", "interstellar/") \
        .replace("CATIE-AQ/", "SmolTalk/") \
        .replace("roborovski/", "roborovski/") \
        .replace("llamafactory/", "llamafactory/") \
        .replace("tejeshbhalla/", "tejeshbhalla/") \
        .replace("Layue13/", "Layue13/")


def render_bar_chart(labels: list[str], values: list[int], output: Path) -> None:
    plt.figure(figsize=(12, 6))
    bars = plt.bar(labels, values, color="#2A9D8F", edgecolor="black")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Example count")
    plt.title("expert-tools dataset composition (10k samples)")
    for bar, count in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{count:,}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()


def render_pie_chart(labels: list[str], values: list[int], output: Path) -> None:
    plt.figure(figsize=(8, 8))
    plt.pie(
        values,
        labels=labels,
        autopct="%1.1f%%",
        startangle=90,
        textprops={"fontsize": 10},
    )
    plt.title("Dataset share by source")
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()


def main() -> None:
    counts = load_counts()
    total = sum(counts.values())
    print("Dataset source distribution:")
    for name, value in counts.items():
        pct = value / total * 100
        print(f"  {name}: {value:,} ({pct:.1f}%)")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    labels = [format_label(name) for name in counts.keys()]
    values = list(counts.values())

    render_bar_chart(labels, values, OUTPUT_DIR / "dataset_distribution.png")
    render_pie_chart(labels, values, OUTPUT_DIR / "dataset_distribution_pie.png")
    print(f"\nCharts saved under {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()

