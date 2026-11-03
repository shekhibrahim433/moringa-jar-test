import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from .config import FIGURES, ROOT

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
          "#e87ba4", "#008300", "#6250d6", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "h"]

INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
NEUTRAL = "#f0efec"
MUTED = "#a9a7a1"

SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#dbe8f8", "#2a78d6", "#0b2f63"])
DIV = LinearSegmentedColormap.from_list("div_blue_orange", ["#2a78d6", NEUTRAL, "#eb6834"])


def apply():
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": INK2,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "axes.axisbelow": True,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "lines.linewidth": 1.2,
        "lines.markersize": 4,
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "axes.prop_cycle": mpl.cycler(color=COLORS),
    })


def panel_label(ax, text):
    ax.set_title(text, loc="left", fontweight="bold")


def save(fig, name, section="eda"):
    out = FIGURES / section
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}")
    return (out / f"{name}.png").relative_to(ROOT).as_posix()
