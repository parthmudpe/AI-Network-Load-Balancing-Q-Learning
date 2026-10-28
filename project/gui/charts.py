"""
charts.py - All Matplotlib charts (one chart is shown at a time).

Each chart is drawn by  draw_chart(figure, chart_name, summary, comparison, metric)
Keeping the drawing code here keeps dashboard.py short and easy to read.
"""
import numpy as np
from matplotlib import colormaps

from algorithms import SHORT_NAMES, AI_NAME

CHART_NAMES = [
    "Requests Handled by Each Server",
    "Server Utilization",
    "Average Response Time",
    "Queue Length",
    "Load Distribution",
    "Algorithm Comparison",
]

# metric name -> (key inside the summary, axis label)
COMPARE_METRICS = {
    "Avg Response Time": ("avg_response", "Average response time (ms)"),
    "Throughput": ("throughput", "Throughput (requests/s)"),
    "Utilization": ("avg_utilization", "Average utilization (%)"),
    "Avg Queue Length": ("avg_queue", "Average queue length (requests)"),
    "Dropped Requests": ("dropped", "Dropped requests"),
}

_PALETTE = colormaps["tab20"]          # 20 distinct colors (25 servers: the last 5 repeat)
ACCENT = "#2563EB"
AI_COLOR = "#16A34A"                   # the AI bar is green in the comparison chart


def server_color(i):
    return _PALETTE(i % 20)
TEXT = "#1F2937"


def draw_chart(fig, chart_name, summary, comparison, metric):
    """Clear the figure and draw the requested chart."""
    fig.clear()
    ax = fig.add_subplot(111)
    ax.set_facecolor("white")
    ax.tick_params(labelsize=10)

    if chart_name == "Algorithm Comparison":
        _comparison_chart(ax, comparison, metric)
    elif summary is None or summary["total"] == 0:
        _empty(ax, "Run a simulation to see this chart")
    elif chart_name == "Requests Handled by Each Server":
        _server_bars(ax, summary, "handled", "Requests handled by each server",
                     "Requests processed", "{:.0f}")
    elif chart_name == "Server Utilization":
        _server_bars(ax, summary, "utilization", "Average server utilization",
                     "Utilization (%)", "{:.1f}%", ymax=100)
    elif chart_name == "Average Response Time":
        _server_bars(ax, summary, "server_response", "Average response time of each server",
                     "Response time (ms)", "{:.0f}")
    elif chart_name == "Queue Length":
        _queue_chart(ax, summary)
    elif chart_name == "Load Distribution":
        _server_bars(ax, summary, "distribution", "Load distribution (share of all requests sent)",
                     "Share of requests (%)", "{:.1f}%")

    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout(pad=2.0)
    if (chart_name == "Queue Length" and summary is not None and summary["total"] > 0
            and len(summary["names"]) > 8):
        fig.subplots_adjust(right=0.80)        # leave room for the legend that sits on the right


def _empty(ax, message):
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=13, color="#6B7280")
    ax.set_xticks([])
    ax.set_yticks([])


def _server_bars(ax, summary, key, title, ylabel, label_format, ymax=None):
    names = summary["names"]
    values = summary[key]
    n = len(names)
    positions = np.arange(n)           # bars are placed by position, so two servers with
                                       # the same name can never be drawn on top of each other
    colors = [server_color(i) for i in range(n)]
    bars = ax.bar(positions, values, color=colors, width=0.6 if n <= 8 else 0.75)
    ax.set_xticks(positions)
    if n <= 6:
        ax.set_xticklabels(names, fontsize=10)
    else:                              # many servers: short, rotated labels
        ax.set_xticklabels(names, rotation=60 if n <= 12 else 90, ha="right" if n <= 12 else "center",
                           fontsize=9 if n <= 12 else 7)
    ax.bar_label(bars, labels=[label_format.format(v) for v in values],
                 fontsize=10 if n <= 8 else (8 if n <= 12 else 6), padding=3,
                 rotation=0 if n <= 12 else 90)
    ax.set_title(title, fontsize=13, fontweight="bold", color=TEXT, pad=12)
    ax.set_xlabel("Server", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    top = ymax if ymax else max(max(values), 1) * (1.15 if n <= 12 else 1.3)
    ax.set_ylim(0, top * (1.12 if ymax and n > 12 else 1))


def _queue_chart(ax, summary):
    times = summary["history_time"]
    history = np.array(summary["history_queue"])
    names = summary["names"]
    if len(times) < 2:
        _empty(ax, "Not enough data yet")
        return
    many = len(names) > 8
    for i, name in enumerate(names):
        ax.plot(times, history[:, i], label=name, linewidth=1.2 if many else 2,
                color=server_color(i), linestyle="--" if i >= 20 else "-")
    ax.set_title("Queue length of each server over time", fontsize=13, fontweight="bold",
                 color=TEXT, pad=12)
    ax.set_xlabel("Simulation time (s)", fontsize=11)
    ax.set_ylabel("Requests waiting in queue", fontsize=11)
    if many:                                   # many lines: legend outside, in columns
        ax.legend(fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0),
                  ncol=2 if len(names) > 13 else 1)
    else:
        ax.legend(fontsize=10, frameon=False, loc="upper right")   # legend needed: several lines


def _comparison_chart(ax, comparison, metric):
    if not comparison:
        _empty(ax, "Open the Comparison tab and click 'Run Comparison'")
        return
    key, ylabel = COMPARE_METRICS[metric]
    labels = [SHORT_NAMES[r["algorithm"]] for r in comparison]
    values = [r["summary"][key] for r in comparison]
    colors = [AI_COLOR if r["algorithm"] == AI_NAME else ACCENT for r in comparison]
    bars = ax.bar(labels, values, color=colors, width=0.55)
    ax.bar_label(bars, labels=["{:.1f}".format(v) if key != "dropped" else "{:.0f}".format(v)
                               for v in values], fontsize=10, padding=3)
    ax.set_title("Algorithm comparison: " + metric, fontsize=13, fontweight="bold",
                 color=TEXT, pad=12)
    ax.set_xlabel("Algorithm", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_ylim(0, max(max(values), 1) * 1.15)
