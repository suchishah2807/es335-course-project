"""One matplotlib style for every figure. Model and baseline colours are fixed project-wide (site uses the same)."""
import matplotlib as mpl
import matplotlib.pyplot as plt

MODEL = "#2a78d6"      # blue: our model (and a single series in audit plots)
BASELINE = "#8a8983"   # neutral grey: baseline / reference
ACCENT2 = "#eb6834"    # orange: second identity (e.g. Failed vs Terminated)
CATS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#b9b8b2"]  # fixed order; last = Other
INK, INK2, GRID = "#1b1b1a", "#55544f", "#e6e5e1"


def apply():
    mpl.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
        "font.family": "DejaVu Sans", "font.size": 9.5,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlesize": 10.5, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.titlecolor": INK, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "xtick.color": INK2, "ytick.color": INK2, "xtick.major.size": 0, "ytick.major.size": 0,
        "legend.frameon": False, "lines.linewidth": 1.8,
    })


def takeaway(fig, text):
    """Grey one-line takeaway under the title area."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    y0 = min(ax.get_tightbbox(r).transformed(fig.transFigure.inverted()).y0 for ax in fig.axes)
    fig.text(0.01, y0 - 0.03, text, ha="left", va="top", fontsize=8.5, color=INK2)


def save(fig, path):
    fig.savefig(path)
    plt.close(fig)
