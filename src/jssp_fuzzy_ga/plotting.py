"""
Plotting helpers. These take result dictionaries produced by
`experiment.run_many` (see notebooks/01_explore.ipynb for how to build them)
rather than reading a hardcoded file -- so you can call these on whatever
results you generate, at whatever N you choose.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OPTIMA = {"ft06": 55, "ft10": 930}
LABELS = {
    "static": "Static (fixed rate)",
    "crisp": "Crisp-adaptive (same rules, hard thresholds)",
    "fuzzy": "Fuzzy-adaptive (this project)",
}
COLORS = {"static": "#888780", "crisp": "#7F77DD", "fuzzy": "#D85A30"}


def plot_convergence(data, instances=("ft06", "ft10"), out_path="convergence_comparison.png"):
    """data[instance][method] = {"curves": (n_runs, n_gens+1) array, "finals": (n_runs,) array}"""
    fig, axes = plt.subplots(1, len(instances), figsize=(6 * len(instances), 4.5))
    if len(instances) == 1:
        axes = [axes]
    for ax, inst in zip(axes, instances):
        for method in ["static", "crisp", "fuzzy"]:
            curves = data[inst][method]["curves"]
            mean_curve = curves.mean(axis=0)
            se_curve = curves.std(axis=0) / np.sqrt(curves.shape[0])
            x = np.arange(curves.shape[1])
            ax.plot(x, mean_curve, label=LABELS[method], color=COLORS[method], linewidth=1.8)
            ax.fill_between(x, mean_curve - se_curve, mean_curve + se_curve, color=COLORS[method], alpha=0.15)
        ax.axhline(OPTIMA.get(inst, 0), color="black", linestyle="--", linewidth=1, label=f"Known optimum ({OPTIMA.get(inst)})")
        ax.set_title(inst.upper())
        ax.set_xlabel("Generation")
        ax.set_ylabel("Best makespan so far")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(fontsize=8, loc="upper right")
    fig.suptitle("Convergence: mean best makespan (shaded = standard error)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    return fig


def plot_final_boxplot(data, instance="ft10", out_path="final_boxplot.png"):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    methods = ["static", "crisp", "fuzzy"]
    box_data = [data[instance][m]["finals"] for m in methods]
    bp = ax.boxplot(box_data, tick_labels=["Static", "Crisp-\nadaptive", "Fuzzy-\nadaptive"], patch_artist=True)
    for patch, method in zip(bp["boxes"], methods):
        patch.set_facecolor(COLORS[method])
        patch.set_alpha(0.5)
    ax.axhline(OPTIMA.get(instance, 0), color="black", linestyle="--", linewidth=1, label=f"Optimum ({OPTIMA.get(instance)})")
    ax.set_ylabel(f"Final best makespan")
    ax.set_title(f"{instance.upper()}: final-makespan distribution by method")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    return fig