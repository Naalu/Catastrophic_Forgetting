"""
Publication-quality visualization for continual learning experiments.

Generates figures for:
1. Loss function comparison (12 experiments)
2. Dropout effect analysis (9 experiments)
3. Depth comparison (3 experiments)
4. Optimizer comparison (9 experiments)
5. Validation curves showing forgetting (3 experiments)

All figures saved as high-resolution PDFs with consistent styling.
No figures are displayed (all saved to disk).

Style: Colorblind-friendly, academic conference standard
References: Clear legends, axis labels, captions ready for LaTeX
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from config import FIGURES_DIR, NUM_TASKS, PlottingStyle
from experiments.runner import ExperimentResult

logger = logging.getLogger(__name__)

# ============================================================================
# PLOTTING UTILITIES
# ============================================================================


def setup_matplotlib_style():
    """
    Configure matplotlib for publication-quality figures.

    Sets:
    - Font families and sizes
    - Line styles and widths
    - Color palettes
    - DPI for high resolution
    """
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica"],
            "font.size": PlottingStyle.FONTSIZE_LABEL,
            "axes.labelsize": PlottingStyle.FONTSIZE_LABEL,
            "axes.titlesize": PlottingStyle.FONTSIZE_TITLE,
            "xtick.labelsize": PlottingStyle.FONTSIZE_TICK,
            "ytick.labelsize": PlottingStyle.FONTSIZE_TICK,
            "legend.fontsize": PlottingStyle.FONTSIZE_LEGEND,
            "lines.linewidth": 2.0,
            "lines.markersize": 6,
            "patch.linewidth": 1.5,
            "axes.linewidth": 1.0,
            "grid.linewidth": 0.5,
            "grid.alpha": 0.3,
        }
    )


def save_figure(fig: plt.Figure, filename: str, tight_layout: bool = True) -> Path:
    """
    Save figure to disk with consistent settings.

    Args:
        fig: Matplotlib figure object
        filename: Filename (without directory or extension)
        tight_layout: Whether to apply tight_layout()

    Returns:
        Path to saved file

    Behavior:
        - Saves to FIGURES_DIR / f"{filename}.pdf"
        - 300 DPI for publication quality
        - Tight layout to minimize whitespace
        - Closes figure after saving (memory management)
    """
    if tight_layout:
        fig.tight_layout()

    filepath = FIGURES_DIR / f"{filename}.pdf"
    fig.savefig(
        filepath,
        dpi=PlottingStyle.DPI,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )

    logger.info(f"Saved figure: {filepath}")
    plt.close(fig)

    return filepath


# ============================================================================
# EXPERIMENT 1: LOSS FUNCTION COMPARISON
# ============================================================================


def plot_loss_comparison(results: Dict[str, ExperimentResult]) -> Path:
    """
    Compare effect of loss functions on ACC, BWT metrics.

    Visualization:
    - 3 subplots (one per depth: 2, 3, 4)
    - Each shows bar chart: [NLL, L1, L2, L1+L2] losses
    - Bars grouped by metric (ACC on left, BWT on right)
    - Colors match PlottingStyle.COLORS['nll', 'l1', 'l2', 'l1_l2']

    Args:
        results: Dictionary of ExperimentResult from loss_comparison suite

    Returns:
        Path to saved figure

    Experiment Design:
        - Fixed: depth varies, dropout=0.0, optimizer=Adam
        - Varying: loss_type ∈ {NLL, L1, L2, L1+L2}
        - Expectation: NLL should perform best (baseline)
        - Analysis: Do L1/L2 regularization help or hurt forgetting?

    Academic Context:
        This addresses: "Effect of various loss functions on forgetting"
        from assignment requirements.
    """

    fig, axes = plt.subplots(1, 3, figsize=PlottingStyle.FIGSIZE_WIDE)
    fig.suptitle(
        "Effect of Loss Function on Catastrophic Forgetting",
        fontsize=PlottingStyle.FONTSIZE_TITLE,
        fontweight="bold",
    )

    depths = [2, 3, 4]
    loss_types = ["nll", "l1", "l2", "l1_l2"]
    loss_labels = ["NLL", "L1", "L2", "L1+L2"]

    for depth_idx, depth in enumerate(depths):
        ax = axes[depth_idx]

        # Collect metrics for this depth
        acc_values = []
        bwt_values = []

        for loss_type in loss_types:
            # Find matching result
            key = f"loss_{loss_type}_depth{depth}"
            if key not in results:
                logger.warning(f"Missing result: {key}")
                acc_values.append(np.nan)
                bwt_values.append(np.nan)
                continue

            result = results[key]
            acc_values.append(result.metrics["acc"])
            bwt_values.append(result.metrics["bwt"])

        # Plot grouped bar chart
        x = np.arange(len(loss_types))
        width = 0.35

        bars1 = ax.bar(
            x - width / 2,
            acc_values,
            width,
            label="ACC",
            color="steelblue",
            alpha=0.8,
            edgecolor="black",
            linewidth=1,
        )
        bars2 = ax.bar(
            x + width / 2,
            bwt_values,
            width,
            label="BWT",
            color="coral",
            alpha=0.8,
            edgecolor="black",
            linewidth=1,
        )

        # Annotations
        ax.set_title(f"Depth = {depth}", fontsize=PlottingStyle.FONTSIZE_LABEL)
        ax.set_ylabel("Metric Value", fontsize=PlottingStyle.FONTSIZE_LABEL)
        ax.set_xticks(x)
        ax.set_xticklabels(loss_labels, fontsize=PlottingStyle.FONTSIZE_TICK)
        ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=PlottingStyle.FONTSIZE_LEGEND)

        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{height:.3f}",
                    ha="center",
                    va="bottom" if height > 0 else "top",
                    fontsize=8,
                )

    return save_figure(fig, "loss_comparison")


# ============================================================================
# EXPERIMENT 2: DROPOUT EFFECT ANALYSIS
# ============================================================================


def plot_dropout_effect(results: Dict[str, ExperimentResult]) -> Path:
    """
    Analyze effect of dropout regularization on forgetting.

    Visualization:
    - 3 subplots (one per depth: 2, 3, 4)
    - Line plot: dropout_rate (x) vs. ACC/BWT (y)
    - Two lines per plot: ACC and BWT
    - Points at dropout ∈ {0.0, 0.2, 0.5}

    Args:
        results: Dictionary from dropout_effect suite

    Returns:
        Path to saved figure

    Experiment Design:
        - Fixed: loss=NLL, optimizer=Adam
        - Varying: dropout_rate ∈ {0.0, 0.2, 0.5}, depth ∈ {2, 3, 4}
        - Expectation: Dropout may hurt forgetting (models less flexible)
        - Analysis: Dropout-forgetting trade-off

    Academic Context:
        This addresses: "Does dropout help? Apply dropout ≤ 0.5"
        from assignment requirements.
    """

    fig, axes = plt.subplots(1, 3, figsize=PlottingStyle.FIGSIZE_WIDE)
    fig.suptitle(
        "Effect of Dropout on Catastrophic Forgetting",
        fontsize=PlottingStyle.FONTSIZE_TITLE,
        fontweight="bold",
    )

    depths = [2, 3, 4]
    dropout_rates = [0.0, 0.2, 0.5]

    for depth_idx, depth in enumerate(depths):
        ax = axes[depth_idx]

        # Collect metrics
        acc_values = []
        bwt_values = []

        for dropout_rate in dropout_rates:
            key = f"dropout_{dropout_rate}_depth{depth}"
            if key not in results:
                logger.warning(f"Missing result: {key}")
                acc_values.append(np.nan)
                bwt_values.append(np.nan)
                continue

            result = results[key]
            acc_values.append(result.metrics["acc"])
            bwt_values.append(result.metrics["bwt"])

        # Convert dropout rates to percentages for x-axis
        x_positions = [dr * 100 for dr in dropout_rates]

        # Plot lines
        ax.plot(
            x_positions,
            acc_values,
            marker="o",
            markersize=8,
            linewidth=2,
            label="ACC",
            color="steelblue",
        )
        ax.plot(
            x_positions,
            bwt_values,
            marker="s",
            markersize=8,
            linewidth=2,
            label="BWT",
            color="coral",
        )

        # Annotations
        ax.set_title(f"Depth = {depth}", fontsize=PlottingStyle.FONTSIZE_LABEL)
        ax.set_xlabel("Dropout Rate (%)", fontsize=PlottingStyle.FONTSIZE_LABEL)
        ax.set_ylabel("Metric Value", fontsize=PlottingStyle.FONTSIZE_LABEL)
        ax.set_xticks(x_positions)
        ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=PlottingStyle.FONTSIZE_LEGEND)

    return save_figure(fig, "dropout_effect")


# ============================================================================
# EXPERIMENT 3: DEPTH ANALYSIS
# ============================================================================


def plot_depth_analysis(results: Dict[str, ExperimentResult]) -> Path:
    """
    Compare performance across network depths.

    Visualization:
    - Single figure, bar chart
    - X-axis: depth ∈ {2, 3, 4}
    - Bars for: ACC, BWT, FWT
    - Heights show metric values
    - Error bars optional (not computing, but space for future)

    Args:
        results: Dictionary from depth_analysis suite

    Returns:
        Path to saved figure

    Experiment Design:
        - Fixed: loss=NLL, dropout=0.0, optimizer=Adam
        - Varying: depth ∈ {2, 3, 4}
        - Expectation: Deeper networks may overfit, hurt BWT
        - Analysis: Depth-forgetting relationship

    Academic Context:
        This addresses: "Effect of depth on forgetting"
        from assignment requirements.
    """

    fig, ax = plt.subplots(figsize=PlottingStyle.FIGSIZE_SINGLE)
    fig.suptitle(
        "Effect of Network Depth on Catastrophic Forgetting",
        fontsize=PlottingStyle.FONTSIZE_TITLE,
        fontweight="bold",
    )

    depths = [2, 3, 4]
    metrics_to_plot = ["acc", "bwt", "fwt"]
    metric_labels = ["ACC", "BWT", "FWT"]
    colors_metrics = ["steelblue", "coral", "seagreen"]

    # Collect metrics
    metrics_data = {m: [] for m in metrics_to_plot}

    for depth in depths:
        key = f"depth_{depth}"
        if key not in results:
            logger.warning(f"Missing result: {key}")
            for m in metrics_to_plot:
                metrics_data[m].append(np.nan)
            continue

        result = results[key]
        for m in metrics_to_plot:
            metrics_data[m].append(result.metrics[m])

    # Plot grouped bars
    x = np.arange(len(depths))
    width = 0.25

    for metric_idx, (metric, metric_label, color) in enumerate(
        zip(metrics_to_plot, metric_labels, colors_metrics)
    ):
        offset = (metric_idx - 1) * width
        bars = ax.bar(
            x + offset,
            metrics_data[metric],
            width,
            label=metric_label,
            color=color,
            alpha=0.8,
            edgecolor="black",
            linewidth=1,
        )

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.3f}",
                ha="center",
                va="bottom" if height > 0 else "top",
                fontsize=9,
            )

    # Annotations
    ax.set_xlabel(
        "Network Depth (Hidden Layers)", fontsize=PlottingStyle.FONTSIZE_LABEL
    )
    ax.set_ylabel("Metric Value", fontsize=PlottingStyle.FONTSIZE_LABEL)
    ax.set_xticks(x)
    ax.set_xticklabels([str(d) for d in depths])
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=PlottingStyle.FONTSIZE_LEGEND, loc="best")

    return save_figure(fig, "depth_analysis")


# ============================================================================
# EXPERIMENT 4: OPTIMIZER COMPARISON
# ============================================================================


def plot_optimizer_comparison(results: Dict[str, ExperimentResult]) -> Path:
    """
    Compare convergence and performance across optimizers.

    Visualization:
    - 3 subplots (one per depth: 2, 3, 4)
    - Bar chart: [SGD, Adam, RMSprop]
    - Bars for: ACC, BWT
    - Colors by optimizer

    Args:
        results: Dictionary from optimizer_comparison suite

    Returns:
        Path to saved figure

    Experiment Design:
        - Fixed: loss=NLL, dropout=0.0, depth varies
        - Varying: optimizer ∈ {SGD, Adam, RMSprop}
        - Expectation: Adam typically converges faster
        - Analysis: Optimizer choice impact on forgetting

    Academic Context:
        This addresses: "Does optimizer play a role in less forgetting?"
        from assignment requirements.
    """

    fig, axes = plt.subplots(1, 3, figsize=PlottingStyle.FIGSIZE_WIDE)
    fig.suptitle(
        "Effect of Optimizer on Catastrophic Forgetting",
        fontsize=PlottingStyle.FONTSIZE_TITLE,
        fontweight="bold",
    )

    depths = [2, 3, 4]
    optimizers = ["sgd", "adam", "rmsprop"]
    optimizer_labels = ["SGD", "Adam", "RMSprop"]
    optimizer_colors = ["#7f7f7f", "#17becf", "#bcbd22"]

    for depth_idx, depth in enumerate(depths):
        ax = axes[depth_idx]

        # Collect metrics
        acc_values = []
        bwt_values = []

        for opt in optimizers:
            key = f"optimizer_{opt}_depth{depth}"
            if key not in results:
                logger.warning(f"Missing result: {key}")
                acc_values.append(np.nan)
                bwt_values.append(np.nan)
                continue

            result = results[key]
            acc_values.append(result.metrics["acc"])
            bwt_values.append(result.metrics["bwt"])

        # Plot grouped bars
        x = np.arange(len(optimizers))
        width = 0.35

        bars1 = ax.bar(
            x - width / 2,
            acc_values,
            width,
            label="ACC",
            color="steelblue",
            alpha=0.8,
            edgecolor="black",
            linewidth=1,
        )
        bars2 = ax.bar(
            x + width / 2,
            bwt_values,
            width,
            label="BWT",
            color="coral",
            alpha=0.8,
            edgecolor="black",
            linewidth=1,
        )

        # Annotations
        ax.set_title(f"Depth = {depth}", fontsize=PlottingStyle.FONTSIZE_LABEL)
        ax.set_ylabel("Metric Value", fontsize=PlottingStyle.FONTSIZE_LABEL)
        ax.set_xticks(x)
        ax.set_xticklabels(optimizer_labels, fontsize=PlottingStyle.FONTSIZE_TICK)
        ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=PlottingStyle.FONTSIZE_LEGEND)

        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{height:.3f}",
                    ha="center",
                    va="bottom" if height > 0 else "top",
                    fontsize=8,
                )

    return save_figure(fig, "optimizer_comparison")


# ============================================================================
# EXPERIMENT 5: VALIDATION CURVES (FORGETTING VISUALIZATION)
# ============================================================================


def plot_validation_curves(results: Dict[str, ExperimentResult]) -> Path:
    """
    Plot accuracy on Task 0 (first task) across task sequence.

    Visualization shows catastrophic forgetting clearly:
    - As more tasks are learned, performance on Task 0 degrades
    - Multiple lines, one per depth (2, 3, 4)
    - X-axis: Task index (0-9, when evaluated)
    - Y-axis: Accuracy on Task 0
    - Downward slope indicates forgetting

    Args:
        results: Dictionary from validation_curves suite

    Returns:
        Path to saved figure

    Experiment Design:
        - Fixed: loss=NLL, dropout=0.0, optimizer=Adam
        - Varying: depth ∈ {2, 3, 4}
        - Captures: Full training history from training_history field
        - Visualization: Single plot with 3 lines (one per depth)

    Academic Context:
        This addresses: "Plot validation results for decrease in model
        prediction when finished training on all 10 tasks" from assignment.

        This is THE key visualization for catastrophic forgetting.
        Shows quantitatively how much performance degrades on old tasks.

    Implementation Notes:
        - training_history contains per-task accuracy records
        - Extract Task 0 accuracy after each task training completes
        - Plot as line to show trajectory
    """

    fig, ax = plt.subplots(figsize=PlottingStyle.FIGSIZE_SINGLE)
    fig.suptitle(
        "Catastrophic Forgetting: Accuracy Degradation on Task 0",
        fontsize=PlottingStyle.FONTSIZE_TITLE,
        fontweight="bold",
    )

    depths = [2, 3, 4]
    depth_colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    for depth, color in zip(depths, depth_colors):
        key = f"validation_depth{depth}"
        if key not in results:
            logger.warning(f"Missing result: {key}")
            continue

        result = results[key]

        # Extract Task 0 accuracy after training each task
        task_0_accuracies = []
        task_indices = []

        for history_entry in result.training_history:
            task_id = history_entry["task"]
            accuracies = history_entry.get(
                "accuracies_after", history_entry.get("accuracies")
            )

            if accuracies is not None:
                task_0_acc = accuracies[0]  # Accuracy on Task 0
                task_0_accuracies.append(task_0_acc)
                task_indices.append(task_id)

        # Plot line
        ax.plot(
            task_indices,
            task_0_accuracies,
            marker="o",
            markersize=8,
            linewidth=2.5,
            label=f"Depth {depth}",
            color=color,
            alpha=0.8,
        )

    # Annotations
    ax.set_xlabel("Task Completed", fontsize=PlottingStyle.FONTSIZE_LABEL)
    ax.set_ylabel("Accuracy on Task 0", fontsize=PlottingStyle.FONTSIZE_LABEL)
    ax.set_xticks(range(NUM_TASKS))
    ax.set_xticklabels([str(i) for i in range(NUM_TASKS)])
    ax.set_ylim([0, 1.0])
    ax.grid(alpha=0.3)
    ax.legend(fontsize=PlottingStyle.FONTSIZE_LEGEND, loc="best")

    # Add annotation for forgetting direction
    ax.annotate(
        "Forgetting",
        xy=(8, 0.1),
        xytext=(6, 0.3),
        arrowprops=dict(arrowstyle="->", color="red", lw=2),
        fontsize=PlottingStyle.FONTSIZE_LEGEND,
        color="red",
        fontweight="bold",
    )

    return save_figure(fig, "validation_curves")


# ============================================================================
# SUMMARY HEATMAPS
# ============================================================================


def plot_task_matrices_heatmap(
    results: Dict[str, ExperimentResult], selected_configs: Optional[List[str]] = None
) -> Path:
    """
    Plot task matrices as heatmaps (one per selected config).

    Visualization:
    - Multiple subplots, one task matrix per subplot
    - Heatmap shows R[i,j] accuracy values
    - Color scale: 0 (white) → 1 (dark blue)
    - Main diagonal (R[i,i]) should be high (immediate performance)
    - Upper triangle shows forward transfer potential
    - Lower triangle shows backward transfer / forgetting

    Args:
        results: Dictionary of ExperimentResult
        selected_configs: List of config names to plot (if None, top 3 by ACC)

    Returns:
        Path to saved figure

    Academic Context:
        Task matrix R is core metric from GEM paper. Visual inspection
        reveals forgetting patterns and transfer structure clearly.
    """

    if selected_configs is None:
        # Select top 3 by ACC
        top_3 = sorted(
            results.items(), key=lambda x: x[1].metrics.get("acc", 0), reverse=True
        )[:3]
        selected_configs = [name for name, _ in top_3]

    num_configs = len(selected_configs)
    fig, axes = plt.subplots(1, num_configs, figsize=(5 * num_configs, 5))

    if num_configs == 1:
        axes = [axes]

    fig.suptitle(
        "Task Matrices: Forward Transfer (Upper Triangle) & Forgetting (Lower Triangle)",
        fontsize=PlottingStyle.FONTSIZE_TITLE,
        fontweight="bold",
    )

    for ax, config_name in zip(axes, selected_configs):
        if config_name not in results:
            logger.warning(f"Missing config: {config_name}")
            continue

        result = results[config_name]
        R = result.task_matrix.R

        # Create heatmap
        im = ax.imshow(R, cmap="Blues", vmin=0, vmax=1, aspect="auto")

        # Add text annotations
        for i in range(R.shape[0]):
            for j in range(R.shape[1]):
                text = ax.text(
                    j,
                    i,
                    f"{R[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="white" if R[i, j] > 0.5 else "black",
                    fontsize=8,
                )

        ax.set_xticks(range(NUM_TASKS))
        ax.set_yticks(range(NUM_TASKS))
        ax.set_xlabel("Task", fontsize=PlottingStyle.FONTSIZE_LABEL)
        ax.set_ylabel("After Training Task", fontsize=PlottingStyle.FONTSIZE_LABEL)
        ax.set_title(config_name, fontsize=PlottingStyle.FONTSIZE_LABEL)

    # Colorbar
    cbar = fig.colorbar(im, ax=axes, orientation="vertical", pad=0.02)
    cbar.set_label("Accuracy", fontsize=PlottingStyle.FONTSIZE_LABEL)

    return save_figure(fig, "task_matrices_heatmap")


# ============================================================================
# SUMMARY TABLE GENERATION
# ============================================================================


def generate_metrics_summary_table(results: Dict[str, ExperimentResult]) -> str:
    """
    Generate ASCII table of all metrics for terminal display.

    Args:
        results: Dictionary of ExperimentResult

    Returns:
        Formatted string with table

    Format:
        Config Name        | ACC    | BWT    | FWT    | TBWT   | CBWT
        ---|---|---|---|---|---
        loss_nll_depth2    | 0.842  |-0.156  | 0.025  | 0.089  | 0.087
        ...
    """

    lines = [
        "=" * 100,
        f"{'Config':<30} | {'ACC':>8} | {'BWT':>8} | {'FWT':>8} | {'TBWT':>8} | {'CBWT':>8}",
        "-" * 100,
    ]

    for config_name in sorted(results.keys()):
        result = results[config_name]
        metrics = result.metrics

        line = (
            f"{config_name:<30} | "
            f"{metrics.get('acc', np.nan):>8.4f} | "
            f"{metrics.get('bwt', np.nan):>8.4f} | "
            f"{metrics.get('fwt', np.nan):>8.4f} | "
            f"{metrics.get('tbwt', np.nan):>8.4f} | "
            f"{metrics.get('cbwt', np.nan):>8.4f}"
        )
        lines.append(line)

    lines.append("=" * 100)

    return "\n".join(lines)


def save_metrics_summary_json(
    results: Dict[str, ExperimentResult], filename: str = "metrics_summary.json"
) -> Path:
    """
    Save all metrics as JSON for external analysis.

    Args:
        results: Dictionary of ExperimentResult
        filename: Output filename

    Returns:
        Path to saved file
    """
    import json

    summary = {}
    for config_name, result in results.items():
        summary[config_name] = {
            "config": result.config.to_dict(),
            "metrics": result.metrics,
            "task_matrix_shape": result.task_matrix.R.shape,
        }

    filepath = FIGURES_DIR / filename
    with open(filepath, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Saved metrics summary: {filepath}")
    return filepath


# ============================================================================
# ADDITIONAL VISUALIZATION FOR ACADEMIC COMPARISON
# ============================================================================


def plot_acc_vs_bwt_scatter(results: Dict[str, ExperimentResult]) -> Path:
    """
    Scatter plot showing ACC vs. BWT trade-off across all experiments.

    Visualization:
    - X-axis: Average Accuracy (ACC)
    - Y-axis: Backward Transfer (BWT)
    - Each point: one experiment
    - Color: experiment suite (loss, dropout, depth, optimizer)
    - Size: forward transfer (FWT) magnitude

    Purpose:
    - Reveals whether higher ACC comes at cost of more forgetting
    - Identifies Pareto-optimal configurations
    - Helps compare to baseline methods (e.g., GEM paper results)

    Args:
        results: Dictionary of all ExperimentResult

    Returns:
        Path to saved figure

    Academic Context:
        GEM paper shows similar scatter plots comparing multiple methods.
        This allows visual assessment of our baseline MLP performance.
    """

    fig, ax = plt.subplots(figsize=PlottingStyle.FIGSIZE_SINGLE)
    fig.suptitle(
        "Performance Trade-off: Accuracy vs. Catastrophic Forgetting",
        fontsize=PlottingStyle.FONTSIZE_TITLE,
        fontweight="bold",
    )

    # Categorize results by experiment type
    categories = {"loss": {}, "dropout": {}, "depth": {}, "optimizer": {}}

    for config_name, result in results.items():
        m = result.metrics
        acc = m.get("acc", np.nan)
        bwt = m.get("bwt", np.nan)
        fwt = m.get("fwt", np.nan)

        # Determine category
        if "loss_" in config_name:
            category = "loss"
        elif "dropout_" in config_name:
            category = "dropout"
        elif "depth_" in config_name:
            category = "depth"
        elif "optimizer_" in config_name:
            category = "optimizer"
        else:
            category = "other"

        if category not in categories:
            categories[category] = {}

        categories[category][config_name] = {"acc": acc, "bwt": bwt, "fwt": fwt}

    # Plot each category with different color and marker
    colors_map = {
        "loss": "#d62728",
        "dropout": "#9467bd",
        "depth": "#2ca02c",
        "optimizer": "#ff7f0e",
    }
    markers_map = {"loss": "o", "dropout": "s", "depth": "^", "optimizer": "D"}

    for category, configs in categories.items():
        if not configs:
            continue

        accs = [cfg["acc"] for cfg in configs.values()]
        bwts = [cfg["bwt"] for cfg in configs.values()]
        fwts = [abs(cfg["fwt"]) for cfg in configs.values()]

        # Scale FWT to point size (5-200)
        sizes = 50 + (np.array(fwts) * 1000)

        ax.scatter(
            accs,
            bwts,
            s=sizes,
            alpha=0.6,
            color=colors_map.get(category, "#999999"),
            marker=markers_map.get(category, "o"),
            edgecolors="black",
            linewidth=1,
            label=category.replace("_", " ").title(),
        )

    # Add reference lines
    ax.axhline(
        y=0,
        color="black",
        linestyle="--",
        linewidth=1,
        alpha=0.5,
        label="Zero forgetting",
    )
    ax.axvline(
        x=0.9, color="gray", linestyle=":", linewidth=1, alpha=0.5, label="90% accuracy"
    )

    # Annotations
    ax.set_xlabel("Average Accuracy (ACC)", fontsize=PlottingStyle.FONTSIZE_LABEL)
    ax.set_ylabel("Backward Transfer (BWT)", fontsize=PlottingStyle.FONTSIZE_LABEL)
    ax.set_xlim([0.5, 1.0])
    ax.set_ylim([-0.7, 0.2])
    ax.grid(alpha=0.3)
    ax.legend(fontsize=PlottingStyle.FONTSIZE_LEGEND, loc="best")

    # Add text annotation for interpretation
    ax.text(
        0.95,
        -0.65,
        "Point size = Forward Transfer (FWT)\nUpper right = Better performance",
        ha="right",
        va="bottom",
        fontsize=9,
        style="italic",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
    )

    return save_figure(fig, "acc_vs_bwt_scatter")


def plot_cumulative_forgetting(results: Dict[str, ExperimentResult]) -> Path:
    """
    Show cumulative forgetting as task sequence progresses.

    Visualization:
    - X-axis: Task completed (0-9)
    - Y-axis: Cumulative accuracy loss (compared to immediate post-training)
    - One line per depth (2, 3, 4)
    - Stacked area or line plot

    Purpose:
    - Shows total "damage" from catastrophic forgetting
    - Visualizes whether depth makes systems more or less forgetful
    - Clear demonstration of the catastrophic forgetting problem

    Args:
        results: Dictionary of ExperimentResult

    Returns:
        Path to saved figure

    Implementation:
        For each validation config (best of each depth):
        - Track accuracy on each task at T=i vs at T=9
        - Sum total loss across all tasks
        - Plot cumulative loss over task sequence
    """

    fig, ax = plt.subplots(figsize=PlottingStyle.FIGSIZE_SINGLE)
    fig.suptitle(
        "Cumulative Accuracy Loss: Severity of Catastrophic Forgetting",
        fontsize=PlottingStyle.FONTSIZE_TITLE,
        fontweight="bold",
    )

    depths = [2, 3, 4]
    depth_colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    for depth, color in zip(depths, depth_colors):
        key = f"validation_depth{depth}"
        if key not in results:
            logging.warning(f"Missing: {key}")
            continue

        result = results[key]

        # Extract per-task accuracies throughout training
        task_accuracies_per_phase = {}  # task_id -> list of accuracies after each training phase

        for history_entry in result.training_history:
            task_trained = history_entry["task"]
            accuracies = history_entry.get(
                "accuracies_after", history_entry.get("accuracies")
            )

            if accuracies is None:
                continue

            # Record accuracy on each task after training task_trained
            for task_id, acc in enumerate(accuracies):
                if task_id not in task_accuracies_per_phase:
                    task_accuracies_per_phase[task_id] = []
                task_accuracies_per_phase[task_id].append(acc)

        # Compute cumulative loss
        cumulative_loss = []

        for phase, task_trained in enumerate(range(NUM_TASKS)):
            # Loss at this phase = sum of degradation on previous tasks
            total_loss = 0
            for prev_task_id in range(task_trained):
                if prev_task_id in task_accuracies_per_phase:
                    accs = task_accuracies_per_phase[prev_task_id]
                    if len(accs) > 0:
                        peak_acc = (
                            accs[prev_task_id] if prev_task_id < len(accs) else accs[-1]
                        )
                        current_acc = accs[-1] if len(accs) > 0 else 0
                        loss = max(0, peak_acc - current_acc)
                        total_loss += loss

            cumulative_loss.append(total_loss)

        ax.plot(
            range(NUM_TASKS),
            cumulative_loss,
            marker="o",
            markersize=8,
            linewidth=2.5,
            label=f"Depth {depth}",
            color=color,
            alpha=0.8,
        )

    # Annotations
    ax.set_xlabel("Task Completed", fontsize=PlottingStyle.FONTSIZE_LABEL)
    ax.set_ylabel("Cumulative Accuracy Loss", fontsize=PlottingStyle.FONTSIZE_LABEL)
    ax.set_xticks(range(NUM_TASKS))
    ax.set_xticklabels([str(i) for i in range(NUM_TASKS)])
    ax.grid(alpha=0.3)
    ax.legend(fontsize=PlottingStyle.FONTSIZE_LEGEND, loc="best")

    return save_figure(fig, "cumulative_forgetting")


def plot_metrics_distribution_boxplots(
    results: Dict[str, ExperimentResult],
    aggregated_by_suite: Dict[str, Dict[str, ExperimentResult]],
) -> Path:
    """
    Box plots showing distribution of metrics across experiment suites.

    Visualization:
    - Separate subplots for ACC, BWT, FWT
    - Each subplot shows 5 boxes (one per experiment suite)
    - Boxes show: median, quartiles, outliers

    Purpose:
    - Statistical comparison of experiment suites
    - Shows consistency/variance within each suite
    - Identifies which experimental choices most affect performance

    Args:
        results: Dictionary of all ExperimentResult
        aggregated_by_suite: Results organized by suite

    Returns:
        Path to saved figure
    """

    fig, axes = plt.subplots(1, 3, figsize=PlottingStyle.FIGSIZE_WIDE)
    fig.suptitle(
        "Distribution of Metrics Across Experiment Suites",
        fontsize=PlottingStyle.FONTSIZE_TITLE,
        fontweight="bold",
    )

    metric_names = ["acc", "bwt", "fwt"]
    metric_labels = ["Average Accuracy", "Backward Transfer", "Forward Transfer"]

    for ax, metric_name, metric_label in zip(axes, metric_names, metric_labels):
        # Collect data per suite
        suite_data = []
        suite_labels = []

        for suite_name, suite_results in aggregated_by_suite.items():
            if not suite_results:
                continue

            metric_values = [
                r.metrics.get(metric_name, np.nan) for r in suite_results.values()
            ]
            suite_data.append(metric_values)
            suite_labels.append(suite_name.replace("_", "\n"))

        # Create box plot
        bp = ax.boxplot(
            suite_data,
            labels=suite_labels,
            patch_artist=True,
            showmeans=True,
            meanprops=dict(marker="D", markerfacecolor="red", markeredgecolor="red"),
        )

        # Color boxes
        colors = ["lightblue", "lightgreen", "lightyellow", "lightcoral", "plum"]
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)

        # Annotations
        ax.set_ylabel(metric_label, fontsize=PlottingStyle.FONTSIZE_LABEL)
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis="x", labelsize=PlottingStyle.FONTSIZE_TICK - 2)

    return save_figure(fig, "metrics_distribution_boxplots")


# ============================================================================
# MASTER PLOTTING FUNCTION
# ============================================================================


def generate_all_visualizations(
    results: Dict[str, ExperimentResult], experiment_suites: Dict[str, List[str]]
) -> Dict[str, Path]:
    """
    [UPDATED] Generate all figures including additional academic comparisons.

    Now includes:
    - Original 5 experiment-specific plots
    - Task matrix heatmaps
    - ACC vs. BWT scatter plot (trade-off analysis)
    - Cumulative forgetting analysis
    - Metrics distribution boxplots
    - Summary tables

    Args:
        results: Dictionary of all ExperimentResult
        experiment_suites: Mapping of suite name → config list

    Returns:
        Dictionary mapping figure name → Path to saved PDF
    """

    setup_matplotlib_style()
    figure_paths = {}

    # Original 5 plots
    logger.info("Generating loss comparison figure...")
    loss_results = {
        k: v
        for k, v in results.items()
        if any(name in k for name in experiment_suites.get("loss_comparison", []))
    }
    if loss_results:
        figure_paths["loss_comparison"] = plot_loss_comparison(loss_results)

    logger.info("Generating dropout effect figure...")
    dropout_results = {
        k: v
        for k, v in results.items()
        if any(name in k for name in experiment_suites.get("dropout_effect", []))
    }
    if dropout_results:
        figure_paths["dropout_effect"] = plot_dropout_effect(dropout_results)

    logger.info("Generating depth analysis figure...")
    depth_results = {
        k: v
        for k, v in results.items()
        if any(name in k for name in experiment_suites.get("depth_analysis", []))
    }
    if depth_results:
        figure_paths["depth_analysis"] = plot_depth_analysis(depth_results)

    logger.info("Generating optimizer comparison figure...")
    opt_results = {
        k: v
        for k, v in results.items()
        if any(name in k for name in experiment_suites.get("optimizer_comparison", []))
    }
    if opt_results:
        figure_paths["optimizer_comparison"] = plot_optimizer_comparison(opt_results)

    logger.info("Generating validation curves figure...")
    val_results = {
        k: v
        for k, v in results.items()
        if any(name in k for name in experiment_suites.get("validation_curves", []))
    }
    if val_results:
        figure_paths["validation_curves"] = plot_validation_curves(val_results)

    # NEW: Academic comparison plots
    logger.info("Generating ACC vs. BWT scatter plot...")
    figure_paths["acc_vs_bwt_scatter"] = plot_acc_vs_bwt_scatter(results)

    logger.info("Generating cumulative forgetting analysis...")
    figure_paths["cumulative_forgetting"] = plot_cumulative_forgetting(results)

    logger.info("Generating task matrix heatmaps...")
    figure_paths["task_matrices"] = plot_task_matrices_heatmap(results)

    # Aggregate results for distribution plots
    aggregated = {}
    for suite_name, configs in experiment_suites.items():
        suite_results = {
            cfg.experiment_name: results[cfg.experiment_name]
            for cfg in configs
            if cfg.experiment_name in results
        }
        if suite_results:
            aggregated[suite_name] = suite_results

    logger.info("Generating metrics distribution boxplots...")
    figure_paths["metrics_distribution"] = plot_metrics_distribution_boxplots(
        results, aggregated
    )

    # Summary tables
    logger.info("Generating metrics summary...")
    metrics_table = generate_metrics_summary_table(results)
    print("\n" + metrics_table)

    figure_paths["metrics_json"] = save_metrics_summary_json(results)

    logger.info(f"All visualizations complete: {len(figure_paths)} figures generated")
    return figure_paths
