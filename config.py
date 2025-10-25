"""
Configuration module for catastrophic forgetting experiments.

This is the SINGLE SOURCE OF TRUTH for all experiment parameters.
All hyperparameters, paths, and experiment definitions are centralized here.

Key Design Principle:
    - Change one place, affects all experiments
    - Easy to modify experiments without touching other code
    - Reproducible and auditable configuration
"""

import hashlib
import json
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List

# ============================================================================
# SEED GENERATION FROM "KARL"
# ============================================================================


def get_seed_value(identifier: str = "Karl") -> int:
    """
    Convert identifier string to deterministic seed value.

    Uses MD5 hashing to ensure reproducibility across Python versions
    and implementations. The hash is deterministic (same input always
    produces same output) and well-distributed across [0, 2^31).

    Args:
        identifier: String to convert to seed (default: "Karl" per assignment)

    Returns:
        Integer seed in range [0, 2^31 - 1] suitable for:
        - np.random.seed()
        - tf.random.set_seed()
        - random.seed()

    Examples:
        >>> seed1 = get_seed_value("Karl")
        >>> seed2 = get_seed_value("Karl")
        >>> seed1 == seed2
        True

        >>> seed_diff = get_seed_value("Different")
        >>> seed1 == seed_diff
        False  # With extremely high probability

    Performance:
        O(1) amortized time (hash computation is constant-time)
        O(n) space where n = len(identifier)
    """
    hash_obj = hashlib.md5(identifier.encode())
    seed = int(hash_obj.hexdigest(), 16) % (2**31)
    return seed


# Generate seed from "Karl" - deterministic across all runs
SEED_VALUE = get_seed_value("Karl")

# ============================================================================
# PERMUTED MNIST PROTOCOL CONSTANTS
# ============================================================================

# Task sequence configuration (per assignment requirements)
NUM_TASKS = 10  # Total number of tasks (0-9)
NUM_CLASSES = 10  # MNIST digit classes (0-9)
INPUT_SHAPE = (784,)  # Flattened 28x28 images

# Training protocol (per assignment)
EPOCHS_TASK_1 = 50  # Initial task: 50 epochs
EPOCHS_PER_TASK = 20  # Each subsequent task: 20 epochs
TOTAL_EPOCHS = EPOCHS_TASK_1 + (NUM_TASKS - 1) * EPOCHS_PER_TASK  # 230 total

# Data characteristics (MNIST standard)
MNIST_TRAIN_SIZE = 60000  # Training samples per task
MNIST_TEST_SIZE = 10000  # Test samples per task
BATCH_SIZE = 32  # Batch size for training

# ============================================================================
# ARCHITECTURE SPECIFICATIONS
# ============================================================================


class ArchitectureDepth(Enum):
    """Valid MLP depths for experiments."""

    DEPTH_2 = 2
    DEPTH_3 = 3
    DEPTH_4 = 4


# List of valid depths for easy iteration
VALID_DEPTHS = [d.value for d in ArchitectureDepth]

# Constant hidden units per layer (per assignment)
HIDDEN_UNITS = 256

# ============================================================================
# LOSS FUNCTION TYPES
# ============================================================================


class LossType(Enum):
    """Available loss functions for comparison."""

    NLL = "nll"  # Negative Log Likelihood (cross-entropy)
    L1 = "l1"  # L1 regularization
    L2 = "l2"  # L2 regularization
    L1_L2 = "l1_l2"  # Combined L1+L2


# List of valid loss types
VALID_LOSSES = [loss.value for loss in LossType]

# Regularization strength for each loss type
REGULARIZATION_STRENGTH = {
    LossType.L1: 1e-4,
    LossType.L2: 1e-4,
    LossType.L1_L2: 1e-4,  # Applied to both L1 and L2
}

# ============================================================================
# DROPOUT RATES FOR REGULARIZATION
# ============================================================================

# Valid dropout rates to test (per assignment: "dropout ≤ 0.5")
VALID_DROPOUT_RATES = [0.0, 0.2, 0.5]

# ============================================================================
# OPTIMIZER CONFIGURATIONS
# ============================================================================


class OptimizerType(Enum):
    """Available optimizers."""

    SGD = "sgd"
    ADAM = "adam"
    RMSPROP = "rmsprop"


# Optimizer-specific hyperparameters (tuned for MNIST)
OPTIMIZER_CONFIG = {
    OptimizerType.SGD: {
        "learning_rate": 0.01,
        "momentum": 0.9,
        "description": "SGD with momentum",
    },
    OptimizerType.ADAM: {
        "learning_rate": 0.001,
        "beta_1": 0.9,
        "beta_2": 0.999,
        "description": "Adam optimizer (adaptive learning rate)",
    },
    OptimizerType.RMSPROP: {
        "learning_rate": 0.001,
        "rho": 0.9,
        "description": "RMSprop (root mean square propagation)",
    },
}

VALID_OPTIMIZERS = [opt.value for opt in OptimizerType]

# ============================================================================
# EXPERIMENT CONFIGURATION DATACLASS
# ============================================================================


@dataclass
class ExperimentConfig:
    """
    Configuration dataclass for a single experiment.

    This class encapsulates all hyperparameters needed to define and execute
    a single experiment. It provides:
    - Validation of parameter values
    - Unique hashing for caching
    - Readable names for filenames
    - Serialization to dictionary

    Key Design Principle:
        Every unique ExperimentConfig maps to exactly one cached result.
        If two configs have identical parameters, they produce the same hash
        and will reuse the cached result (avoiding recomputation).

    Attributes:
        depth (int): Number of hidden layers (2, 3, or 4)
            - Affects model capacity and training time
            - Typically: deeper → slower training, potentially higher accuracy

        loss_type (str): Loss function to use (NLL, L1, L2, L1+L2)
            - NLL: Standard categorical cross-entropy (baseline)
            - L1: Cross-entropy + L1 regularization
            - L2: Cross-entropy + L2 regularization
            - L1_L2: Combined L1 and L2 regularization

        dropout_rate (float): Dropout probability (0.0, 0.2, or 0.5)
            - 0.0: No dropout (potential overfitting on some tasks)
            - 0.2: Light regularization
            - 0.5: Heavy regularization

        optimizer (str): Optimizer algorithm (sgd, adam, rmsprop)
            - SGD: Stochastic gradient descent with momentum
            - Adam: Adaptive moment estimation (default for many tasks)
            - RMSprop: Root mean square propagation

        experiment_name (str): Human-readable identifier
            - Used in results dictionary keys
            - Should be descriptive (e.g., "loss_nll_depth2")

        description (str): Optional detailed description
            - For documentation purposes

    Example:
        >>> config = ExperimentConfig(
        ...     depth=2,
        ...     loss_type="nll",
        ...     dropout_rate=0.0,
        ...     optimizer="adam",
        ...     experiment_name="baseline_nll"
        ... )
        >>> config.get_config_hash()
        'a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6'  # Unique identifier
        >>> config.get_readable_name()
        'depth2_nll_dropout0p0_adam'  # For filenames

    Validation:
        __post_init__() is called automatically after instantiation.
        Raises AssertionError if any parameter is invalid.
    """

    depth: int
    loss_type: str
    dropout_rate: float
    optimizer: str
    experiment_name: str = "default"
    description: str = ""

    def __post_init__(self):
        """
        Validate configuration on instantiation.

        Checks that all parameters are within valid ranges.
        Raises AssertionError if validation fails.
        """
        assert self.depth in VALID_DEPTHS, f"Depth {self.depth} not in {VALID_DEPTHS}"
        assert self.loss_type in VALID_LOSSES, (
            f"Loss {self.loss_type} not in {VALID_LOSSES}"
        )
        assert self.dropout_rate in VALID_DROPOUT_RATES, (
            f"Dropout {self.dropout_rate} not in {VALID_DROPOUT_RATES}"
        )
        assert self.optimizer in VALID_OPTIMIZERS, (
            f"Optimizer {self.optimizer} not in {VALID_OPTIMIZERS}"
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for JSON serialization.

        Returns:
            Dictionary representation of config
        """
        return asdict(self)

    def get_config_hash(self) -> str:
        """
        Generate unique hash for this configuration.

        Used as key for caching results. Two configs with identical
        parameters will produce the same hash.

        Returns:
            Hex string of MD5 hash (32 chars)

        Example:
            >>> config1 = ExperimentConfig(depth=2, loss_type="nll", ...)
            >>> config2 = ExperimentConfig(depth=2, loss_type="nll", ...)
            >>> config1.get_config_hash() == config2.get_config_hash()
            True  # Same parameters = same hash = cache hit
        """
        config_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()

    def get_readable_name(self) -> str:
        """
        Generate human-readable filename for this config.

        Example: "depth2_nll_dropout0p0_adam"

        Returns:
            Slug-formatted string suitable for filenames
        """
        dropout_str = f"dropout{str(self.dropout_rate).replace('.', 'p')}"
        return f"depth{self.depth}_{self.loss_type}_{dropout_str}_{self.optimizer}"


# ============================================================================
# EXPERIMENT MATRICES - WHICH EXPERIMENTS TO RUN
# ============================================================================


def build_experiment_suite() -> Dict[str, List[ExperimentConfig]]:
    """
    Build the complete suite of experiments to execute.

    This function generates all experiment configurations organized by suite.
    Each suite tests a specific hypothesis about catastrophic forgetting.

    Returns:
        Dictionary mapping experiment suite name to list of configs.

    Structure:
        {
            "loss_comparison": [...],      # Exp 1: 12 configs (4 losses × 3 depths)
            "dropout_effect": [...],       # Exp 2: 9 configs (3 dropouts × 3 depths)
            "depth_analysis": [...],       # Exp 3: 3 configs (3 depths)
            "optimizer_comparison": [...], # Exp 4: 9 configs (3 optimizers × 3 depths)
            "validation_curves": [...]     # Exp 5: 3 configs (validation tracking)
        }

    Total: 36 unique experiments (plus validation)

    Academic Context:
        Each suite corresponds to a specific research question:
        1. Does loss function affect forgetting?
        2. Does dropout prevent forgetting?
        3. Does network depth affect forgetting?
        4. Does optimizer choice matter for continual learning?
        5. How does performance degrade over task sequence?
    """

    # ==== EXPERIMENT 1: Loss Function Effect ====
    # Question: Does the loss function affect catastrophic forgetting?
    # Fixed: depth varies, dropout=0.0, optimizer=Adam
    # Varying: loss_type ∈ {NLL, L1, L2, L1+L2}
    loss_comparison = [
        ExperimentConfig(
            depth=d,
            loss_type=loss,
            dropout_rate=0.0,
            optimizer="adam",
            experiment_name=f"loss_{loss}_depth{d}",
            description=f"Loss={loss}, Depth={d}, baseline config",
        )
        for d in VALID_DEPTHS
        for loss in VALID_LOSSES
    ]

    # ==== EXPERIMENT 2: Dropout Effect ====
    # Question: Does dropout regularization help prevent forgetting?
    # Fixed: loss=NLL, optimizer=Adam
    # Varying: dropout_rate ∈ {0.0, 0.2, 0.5}, depth ∈ {2, 3, 4}
    dropout_effect = [
        ExperimentConfig(
            depth=d,
            loss_type="nll",
            dropout_rate=dr,
            optimizer="adam",
            experiment_name=f"dropout_{dr}_depth{d}",
            description=f"Dropout={dr}, Depth={d}, NLL loss",
        )
        for d in VALID_DEPTHS
        for dr in VALID_DROPOUT_RATES
    ]

    # ==== EXPERIMENT 3: Depth Analysis ====
    # Question: Does network depth affect catastrophic forgetting?
    # Fixed: loss=NLL, dropout=0.0, optimizer=Adam
    # Varying: depth ∈ {2, 3, 4}
    depth_analysis = [
        ExperimentConfig(
            depth=d,
            loss_type="nll",
            dropout_rate=0.0,
            optimizer="adam",
            experiment_name=f"depth_{d}",
            description=f"Depth={d}, standard config",
        )
        for d in VALID_DEPTHS
    ]

    # ==== EXPERIMENT 4: Optimizer Comparison ====
    # Question: Does optimizer choice affect continual learning?
    # Fixed: loss=NLL, dropout=0.0
    # Varying: optimizer ∈ {SGD, Adam, RMSprop}, depth ∈ {2, 3, 4}
    optimizer_comparison = [
        ExperimentConfig(
            depth=d,
            loss_type="nll",
            dropout_rate=0.0,
            optimizer=opt,
            experiment_name=f"optimizer_{opt}_depth{d}",
            description=f"Optimizer={opt}, Depth={d}",
        )
        for d in VALID_DEPTHS
        for opt in ["sgd", "adam", "rmsprop"]
    ]

    # ==== EXPERIMENT 5: Validation Curves ====
    # Question: How does performance on Task 0 degrade over task sequence?
    # Uses best performing configs from above experiments
    # Fixed: loss=NLL, dropout=0.0, optimizer=Adam
    # Varying: depth ∈ {2, 3, 4}
    validation_curves = [
        ExperimentConfig(
            depth=d,
            loss_type="nll",
            dropout_rate=0.0,
            optimizer="adam",
            experiment_name=f"validation_depth{d}",
            description=f"Full validation tracking for depth={d}",
        )
        for d in VALID_DEPTHS
    ]

    return {
        "loss_comparison": loss_comparison,
        "dropout_effect": dropout_effect,
        "depth_analysis": depth_analysis,
        "optimizer_comparison": optimizer_comparison,
        "validation_curves": validation_curves,
    }


# ============================================================================
# METRICS & COMPUTATION CONSTANTS
# ============================================================================

# Task matrix dimensions (for clarity in metrics computation)
TASK_MATRIX_SHAPE = (NUM_TASKS, NUM_TASKS)

# Metric names (for consistency across results)
METRIC_NAMES = {
    "acc": "Average Accuracy",
    "bwt": "Backward Transfer",
    "fwt": "Forward Transfer",
    "tbwt": "Task-Based BWT (bonus)",
    "cbwt": "Class-Based BWT (bonus)",
}

# ============================================================================
# VISUALIZATION CONSTANTS
# ============================================================================


class PlottingStyle:
    """
    Publication-ready plot styling constants.

    Provides consistent styling across all figures:
    - Figure sizes optimized for publications
    - Font sizes for readability
    - Colorblind-friendly color palette
    - High-resolution output (300 DPI)
    """

    # Figure sizes (inches, optimized for 2-column papers)
    FIGSIZE_SINGLE = (6, 4)  # Single-column figure
    FIGSIZE_WIDE = (12, 4)  # Two-column figure
    FIGSIZE_SQUARE = (8, 8)  # Square figure (heatmaps)

    # Font sizes (points)
    FONTSIZE_TITLE = 14  # Figure title
    FONTSIZE_LABEL = 12  # Axis labels
    FONTSIZE_TICK = 10  # Tick labels
    FONTSIZE_LEGEND = 10  # Legend font

    # Colors (colorblind-friendly palette)
    COLORS = {
        "depth2": "#1f77b4",  # Blue
        "depth3": "#ff7f0e",  # Orange
        "depth4": "#2ca02c",  # Green
        "nll": "#d62728",  # Red
        "l1": "#9467bd",  # Purple
        "l2": "#8c564b",  # Brown
        "l1_l2": "#e377c2",  # Pink
        "sgd": "#7f7f7f",  # Gray
        "adam": "#17becf",  # Cyan
        "rmsprop": "#bcbd22",  # Yellow-green
    }

    # DPI for saved figures (publication quality)
    DPI = 300

    # File format
    SAVE_FORMAT = "pdf"


# ============================================================================
# DIRECTORY STRUCTURE
# ============================================================================

# Base directories
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
CACHE_DIR = RESULTS_DIR / "cache"
MODELS_DIR = RESULTS_DIR / "models"

# Subdirectories for caching
PERMUTED_MNIST_CACHE = CACHE_DIR / "permuted_mnist"
EXPERIMENT_CACHE = CACHE_DIR / "experiments"

# Create directories if not exist
for directory in [
    DATA_DIR,
    RESULTS_DIR,
    FIGURES_DIR,
    CACHE_DIR,
    MODELS_DIR,
    PERMUTED_MNIST_CACHE,
    EXPERIMENT_CACHE,
]:
    directory.mkdir(parents=True, exist_ok=True)

# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================


def print_experiment_summary() -> None:
    """
    Print a summary of the experiment suite to the console.

    Useful for understanding what experiments will be run.
    """
    suite = build_experiment_suite()

    print("\n" + "=" * 70)
    print("EXPERIMENT SUITE SUMMARY")
    print("=" * 70)

    total_configs = 0
    for suite_name, configs in suite.items():
        print(f"\n{suite_name.upper().replace('_', ' ')}:")
        print(f"  Count: {len(configs)} experiments")

        # Show first and last config as examples
        if configs:
            print("  Examples:")
            print(f"    - {configs[0].experiment_name}")
            if len(configs) > 1:
                print(f"    - {configs[-1].experiment_name}")

        total_configs += len(configs)

    print(f"\n{'=' * 70}")
    print(f"TOTAL: {total_configs} experiments")
    print("Estimated time (first run, GPU): 45-60 minutes")
    print("Estimated time (first run, CPU): 3-5 hours")
    print("Subsequent runs (cache): ~30 seconds")
    print("=" * 70 + "\n")
