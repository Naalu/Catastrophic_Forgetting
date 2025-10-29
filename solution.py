"""
Main orchestration script for catastrophic forgetting analysis.

Executes the complete experimental pipeline:
1. Load experiment configuration suite
2. Run all experiments (with caching)
3. Generate visualizations
4. Produce summary statistics
5. Prepare LaTeX-ready results table

Entry point: python solution.py

Run times:
- First execution: ~5-10 minutes (runs all 42 experiments)
- Subsequent: ~30 seconds (cache hits)

Output:
- Figures: results/figures/*.pdf
- Data: results/cache/experiments/*.json
- Summary: Console printout + JSON
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import tensorflow as tf

# Import our modules
from config import (
    FIGURES_DIR,
    RESULTS_DIR,
    SEED_VALUE,
    ExperimentConfig,
    build_experiment_suite,
)
from experiments.runner import ExperimentResult, ExperimentRunner
from utils.plotting import generate_all_visualizations, generate_metrics_summary_table

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================


def setup_logging(verbose: bool = True) -> None:
    """
    Configure logging for experiment execution.

    Args:
        verbose: If True, show DEBUG level logs; else INFO only
    """
    log_level = logging.DEBUG if verbose else logging.INFO

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(console_formatter)

    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.addHandler(console_handler)

    # Suppress verbose TensorFlow logging
    logging.getLogger("tensorflow").setLevel(logging.WARNING)

    logging.info(f"Logging configured at level {logging.getLevelName(log_level)}")


# ============================================================================
# STARTUP VERIFICATION
# ============================================================================


def verify_gpu_availability() -> str:
    """
    Verify GPU availability and configuration.

    Returns:
        String describing GPU status
    """

    logger = logging.getLogger(__name__)
    gpus = tf.config.list_physical_devices("GPU")

    if gpus:
        logger.info(f"✓ GPU detected: {len(gpus)} GPU(s)")
        for gpu in gpus:
            logger.info(f"  - {gpu}")

        # Set memory growth to avoid OOM
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        return f"Using {len(gpus)} GPU(s)"
    else:
        logger.warning("✗ No GPU detected - using CPU only")
        logger.warning("   On Mac: ensure tensorflow-metal is installed")
        logger.warning("   pip install tensorflow-metal")
        return "CPU only"


# ============================================================================
# EXPERIMENT EXECUTION
# ============================================================================


def run_all_experiments(
    runner: ExperimentRunner, force_regenerate: bool = False, verbose: int = 0
) -> Dict[str, ExperimentResult]:
    """
    Execute all experiments in the suite.

    Args:
        runner: ExperimentRunner instance
        force_regenerate: If True, ignore cache and re-run all
        verbose: TensorFlow verbosity (0=silent, 1=progress)

    Returns:
        Dictionary mapping config name → ExperimentResult

    Orchestration:
        1. Build complete experiment suite (~42 configs)
        2. Run each config (with cache checking)
        3. Track completion time and results
        4. Return aggregated results dict

    Performance:
        First run: 5-10 minutes (all 42 experiments)
        Cached: ~30 seconds (all cache hits)
    """

    logging.info("=" * 70)
    logging.info("BUILDING EXPERIMENT SUITE")
    logging.info("=" * 70)

    experiment_suite = build_experiment_suite()
    total_configs = sum(len(configs) for configs in experiment_suite.values())

    logging.info(f"Total experiments to run: {total_configs}")
    for suite_name, configs in experiment_suite.items():
        logging.info(f"  - {suite_name}: {len(configs)} experiments")

    # Flatten to list
    all_configs = []
    for configs in experiment_suite.values():
        all_configs.extend(configs)

    logging.info("=" * 70)
    logging.info("RUNNING EXPERIMENTS")
    logging.info("=" * 70)

    # Run all experiments
    results = runner.run_experiment_suite(all_configs, verbose=verbose)

    logging.info("=" * 70)
    logging.info("EXPERIMENT EXECUTION COMPLETE")
    logging.info("=" * 70)
    logging.info(f"Completed: {len(results)} experiments")

    return results, experiment_suite


# ============================================================================
# RESULTS AGGREGATION & ANALYSIS
# ============================================================================


def aggregate_results_by_suite(
    results: Dict[str, ExperimentResult],
    experiment_suite: Dict[str, List[ExperimentConfig]],
) -> Dict[str, Dict[str, ExperimentResult]]:
    """
    Organize results by experiment suite.

    Args:
        results: Flat dictionary of all results
        experiment_suite: Suite mapping from build_experiment_suite()

    Returns:
        Nested dictionary organized by suite

        Example:
        {
            "loss_comparison": {
                "loss_nll_depth2": ExperimentResult(...),
                "loss_nll_depth3": ExperimentResult(...),
                ...
            },
            "dropout_effect": {...},
            ...
        }
    """

    aggregated = {}

    for suite_name, configs in experiment_suite.items():
        suite_results = {}
        for config in configs:
            if config.experiment_name in results:
                suite_results[config.experiment_name] = results[config.experiment_name]
        aggregated[suite_name] = suite_results

    return aggregated


def print_results_summary(
    results: Dict[str, ExperimentResult],
    aggregated_by_suite: Dict[str, Dict[str, ExperimentResult]],
) -> None:
    """
    Print comprehensive results summary to console.

    Args:
        results: All results
        aggregated_by_suite: Results organized by suite
    """

    print("\n" + "=" * 80)
    print("COMPREHENSIVE RESULTS SUMMARY")
    print("=" * 80)

    # Overall statistics
    all_accs = [r.metrics.get("acc", np.nan) for r in results.values()]
    all_bwts = [r.metrics.get("bwt", np.nan) for r in results.values()]

    print(f"\nOverall Statistics ({len(results)} experiments):")
    print(f"  ACC: {np.nanmean(all_accs):.4f} ± {np.nanstd(all_accs):.4f}")
    print(f"       Range: [{np.nanmin(all_accs):.4f}, {np.nanmax(all_accs):.4f}]")
    print(f"  BWT: {np.nanmean(all_bwts):.4f} ± {np.nanstd(all_bwts):.4f}")
    print(f"       Range: [{np.nanmin(all_bwts):.4f}, {np.nanmax(all_bwts):.4f}]")

    # Per-suite analysis
    for suite_name, suite_results in aggregated_by_suite.items():
        if not suite_results:
            continue

        suite_accs = [r.metrics.get("acc", np.nan) for r in suite_results.values()]
        suite_bwts = [r.metrics.get("bwt", np.nan) for r in suite_results.values()]

        print(f"\n{suite_name.upper()} ({len(suite_results)} experiments):")
        print(
            f"  ACC: {np.nanmean(suite_accs):.4f} (range: [{np.nanmin(suite_accs):.4f}, {np.nanmax(suite_accs):.4f}])"
        )
        print(
            f"  BWT: {np.nanmean(suite_bwts):.4f} (range: [{np.nanmin(suite_bwts):.4f}, {np.nanmax(suite_bwts):.4f}])"
        )

        # Best and worst by BWT (most important metric for forgetting)
        best_config = min(
            suite_results.items(), key=lambda x: x[1].metrics.get("bwt", np.inf)
        )
        worst_config = max(
            suite_results.items(), key=lambda x: x[1].metrics.get("bwt", -np.inf)
        )

        print(
            f"  Best (least forgetting):  {best_config[0]} (BWT={best_config[1].metrics['bwt']:.4f})"
        )
        print(
            f"  Worst (most forgetting):  {worst_config[0]} (BWT={worst_config[1].metrics['bwt']:.4f})"
        )

    print("\n" + "=" * 80)


def generate_latex_results_table(
    aggregated_by_suite: Dict[str, Dict[str, ExperimentResult]], output_file: Path
) -> Path:
    """
    Generate LaTeX table with results for inclusion in paper.

    Args:
        aggregated_by_suite: Results organized by suite
        output_file: Path to save .tex file

    Returns:
        Path to saved LaTeX file

    Output Format:
        Standard academic table with:
        - Config name
        - ACC, BWT, FWT metrics
        - Ready to \input{} into main LaTeX document
    """

    lines = [
        "% Auto-generated results table",
        "\\begin{table}[t]",
        "\\centering",
        "\\begin{tabular}{l|ccc|ccc}",
        "\\toprule",
        "\\textbf{Configuration} & \\textbf{ACC} & \\textbf{BWT} & \\textbf{FWT} & "
        "\\textbf{TBWT} & \\textbf{CBWT} \\\\",
        "\\midrule",
    ]

    for suite_name, suite_results in aggregated_by_suite.items():
        lines.append(
            f"\\multicolumn{{5}}{{|l|}}{{\\textbf{{{suite_name.replace('_', ' ').title()}}}}} \\\\"
        )

        for config_name in sorted(suite_results.keys()):
            result = suite_results[config_name]
            m = result.metrics

            # Use raw string or double backslash to escape for LaTeX
            config_name_escaped = config_name.replace("_", r"\_")

            line = (
                f"{config_name_escaped:<40} & "
                f"{m.get('acc', np.nan):>6.4f} & "
                f"{m.get('bwt', np.nan):>6.4f} & "
                f"{m.get('fwt', np.nan):>6.4f} & "
                f"{m.get('tbwt', np.nan):>6.4f} & "
                f"{m.get('cbwt', np.nan):>6.4f} \\\\"
            )
            lines.append(line)

        lines.append("\\midrule")

    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{Comprehensive results from catastrophic forgetting analysis.}",
            "\\label{tab:results}",
            "\\end{table}",
        ]
    )

    with open(output_file, "w") as f:
        f.write("\n".join(lines))

    logging.info(f"Generated LaTeX table: {output_file}")
    return output_file


# ============================================================================
# MAIN EXECUTION
# ============================================================================


def main(
    force_regenerate: bool = False, verbose_tf: int = 0, verbose_logging: bool = True
) -> int:
    """
    Main entry point for complete experimental pipeline.

    Args:
        force_regenerate: If True, ignore all caches and re-run everything
        verbose_tf: TensorFlow verbosity (0=silent, 1=progress, 2=detailed)
        verbose_logging: If True, show DEBUG logs; else INFO only

    Returns:
        Exit code (0=success, 1=failure)

    Pipeline:
        1. Setup logging
        2. Create ExperimentRunner
        3. Run all experiments
        4. Aggregate results
        5. Generate visualizations
        6. Print summaries
        7. Save results to disk

    Output Structure:
        results/
        ├── cache/
        │   ├── permuted_mnist/
        │   │   ├── task_00.pkl
        │   │   └── ...
        │   └── experiments/
        │       ├── result_<hash1>.json
        │       └── ...
        ├── figures/
        │   ├── loss_comparison.pdf
        │   ├── dropout_effect.pdf
        │   ├── depth_analysis.pdf
        │   ├── optimizer_comparison.pdf
        │   ├── validation_curves.pdf
        │   ├── task_matrices_heatmap.pdf
        │   └── metrics_summary.json
        └── results_table.tex
    """

    try:
        # Setup
        setup_logging(verbose=verbose_logging)
        gpu_status = verify_gpu_availability()
        logging.info(f"Acceleration: {gpu_status}")
        logging.info("Starting catastrophic forgetting analysis pipeline")
        logging.info(f"Seed value: {SEED_VALUE}")
        logging.info(f"Force regenerate: {force_regenerate}")

        # ============================================================================
        # PATCH: Configure GPU for Metal (M1/M2)
        # ============================================================================
        logging.info("Configuring GPU...")
        gpus = tf.config.list_physical_devices("GPU")

        if gpus:
            logging.info(f"✓ GPU available: {len(gpus)} GPU(s)")
            for gpu in gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    logging.info("  ✓ Memory growth enabled")
                except RuntimeError as e:
                    logging.warning(f"  ✗ Could not set memory growth: {e}")
        else:
            logging.warning("⚠️  No GPU detected - training will use CPU")
        # ============================================================================

        # Create runner
        runner = ExperimentRunner(seed=SEED_VALUE, force_regenerate=force_regenerate)

        # Run experiments
        results_tuple = run_all_experiments(
            runner, force_regenerate=force_regenerate, verbose=verbose_tf
        )
        results, experiment_suite = results_tuple

        if not results:
            logging.error("No results produced!")
            return 1

        # Aggregate results
        aggregated = aggregate_results_by_suite(results, experiment_suite)

        # Print summary
        print_results_summary(results, aggregated)

        # Print detailed metrics table
        print("\n" + generate_metrics_summary_table(results))

        # Generate visualizations
        logging.info("=" * 70)
        logging.info("GENERATING VISUALIZATIONS")
        logging.info("=" * 70)

        experiment_suite_mapping = {
            suite_name: [cfg.experiment_name for cfg in configs]
            for suite_name, configs in experiment_suite.items()
        }

        figure_paths = generate_all_visualizations(results, experiment_suite_mapping)

        logging.info("=" * 70)
        logging.info("GENERATING LATEX TABLE")
        logging.info("=" * 70)

        latex_table_path = generate_latex_results_table(
            aggregated, RESULTS_DIR / "results_table.tex"
        )

        # Final summary
        logging.info("=" * 70)
        logging.info("PIPELINE COMPLETE")
        logging.info("=" * 70)
        logging.info(f"Results saved to: {RESULTS_DIR}")
        logging.info(f"Figures saved to: {FIGURES_DIR}")
        logging.info(f"LaTeX table saved to: {latex_table_path}")

        return 0

    except Exception as e:
        logging.error(f"Pipeline failed with error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    # Parse command-line arguments
    import argparse

    parser = argparse.ArgumentParser(
        description="Catastrophic Forgetting Analysis: CS 599 Deep Learning"
    )
    parser.add_argument(
        "--force-regenerate",
        action="store_true",
        help="Ignore all caches and re-run experiments",
    )
    parser.add_argument(
        "--verbose-tf",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="TensorFlow verbosity level",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress DEBUG logging")

    args = parser.parse_args()

    exit_code = main(
        force_regenerate=args.force_regenerate,
        verbose_tf=args.verbose_tf,
        verbose_logging=not args.quiet,
    )

    sys.exit(exit_code)
