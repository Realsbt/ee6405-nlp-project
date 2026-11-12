"""Render the implemented gated variant, without implying measured improvement."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def render(output):
    figure, axis = plt.subplots(figsize=(14, 6.2))
    axis.set(xlim=(0, 14), ylim=(0, 6.2))
    axis.axis("off")

    def box(x, y, width, height, title, detail, new=False):
        axis.add_patch(FancyBboxPatch(
            (x, y), width, height, boxstyle="round,pad=0.03,rounding_size=0.06",
            facecolor="#E3F2EC" if new else "#EDF0F2",
            edgecolor="#19846B" if new else "#637580", linewidth=1.4,
        ))
        axis.text(x + width / 2, y + height * 0.66, title,
                  ha="center", va="center", fontsize=11, weight="bold")
        axis.text(x + width / 2, y + height * 0.29, detail,
                  ha="center", va="center", fontsize=9, color="#394A50")

    def arrow(start, end, dashed=False, via=None):
        axis.add_patch(FancyArrowPatch(
            start, end, arrowstyle="-|>", mutation_scale=13,
            linewidth=1.5, linestyle="--" if dashed else "-",
            color="#526269", connectionstyle=via or "arc3,rad=0",
        ))

    axis.text(0.1, 5.85, "CLS-preserving EDU fusion", fontsize=22, weight="bold")
    axis.text(0.1, 5.4, "Implemented gated variant | shared encoder | no relation labels or GNN",
              fontsize=11, color="#526269")
    box(0.1, 2.9, 1.6, 1.1, "Same window", "Up to 512 tokens")
    box(2.2, 2.9, 1.9, 1.1, "RoBERTa", "Base-pretrained")
    box(4.7, 4.1, 1.5, 0.9, "CLS", "Global context")
    box(4.7, 1.7, 1.5, 1.1, "EDU means", "Cached spans")
    box(6.7, 1.7, 1.8, 1.1, "Attention", "Weighted context", new=True)
    box(9.0, 1.7, 1.8, 1.1, "Scalar gate", "sigmoid([CLS; c])", new=True)
    box(9.0, 4.1, 1.8, 0.9, "Residual sum", "CLS + g * c", new=True)
    box(11.4, 4.1, 2.2, 0.9, "Matched head", "Projection + 4 classes")
    arrow((1.73, 3.45), (2.17, 3.45))
    arrow((4.13, 3.7), (4.67, 4.4))
    arrow((4.13, 3.15), (4.67, 2.5))
    arrow((6.23, 4.55), (8.97, 4.55))
    arrow((6.23, 2.25), (6.67, 2.25))
    arrow((8.53, 2.25), (8.97, 2.25))
    arrow((9.9, 2.83), (9.9, 4.07))
    axis.text(10.03, 3.45, "g * c", fontsize=10, va="center", color="#526269")
    arrow((10.83, 4.55), (11.37, 4.55))
    arrow((6.2, 4.18), (9.45, 2.83), dashed=True)
    axis.text(11.1, 3.25, "Same head shape\nin all four variants",
              fontsize=9, color="#526269")
    axis.text(0.1, 0.95, "Ablations", fontsize=11, weight="bold")
    axis.text(1.3, 0.95, "CLS only  /  CLS + mean  /  CLS + attention  /  CLS + gated attention",
              fontsize=11)
    axis.text(0.1, 0.45, "Gray: reused foundation or matched comparator. Green: new fusion modules.",
              fontsize=10, color="#526269")
    axis.text(0.1, 0.08, "Gate starts near 0.119. EDU weights are not causal explanations. Performance requires measured results.",
              fontsize=9, color="#526269")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    render(parser.parse_args().output)
