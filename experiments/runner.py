"""
Experiment runner and orchestration.

Handles:
- Configuration management
- Caching of experiment results
- Sequential task learning protocol
- Result persistence and loading
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from config import (
    BATCH_SIZE,
    EPOCHS_PER_TASK,
    EPOCHS_TASK_1,
    EXPERIMENT_CACHE,
    NUM_TASKS,
    SEED_VALUE,
    ExperimentConfig,
)
from models.mlp import MLPClassifier
from utils.data_pipeline import load_all_tasks
from utils.metrics import TaskMatrix, compute_all_metrics

logger = logging.getLogger(__name__)

# ============================================================================
# EXPERIMENT RESULT STORAGE
# ============================================================================


class ExperimentResult:
    """
    Encapsulates results from a single experiment.

    Attributes:
        config: ExperimentConfig that produced this result
        task_matrix: TaskMatrix with all accuracy scores
        metrics: Dictionary of computed metrics (ACC, BWT, etc.)
        training_history: Per-epoch training metrics
        model_weights: Final model weights (optional, for disk savings)
    """

    def __init__(self, config: ExperimentConfig):
        """Initialize experiment result."""
        self.config = config
        self.task_matrix = TaskMatrix(NUM_TASKS)
        self.metrics = {}
        self.training_history = []
        self.model_weights = None

    def to_dict(self) -> Dict:
        """
        Convert to dictionary for JSON serialization.

        Note: numpy arrays are converted to lists.
        """
        return {
            "config": self.config.to_dict(),
            "task_matrix": self.task_matrix.R.tolist(),
            "metrics": self.metrics,
            "training_history": self.training_history,
        }

    @staticmethod
    def from_dict(data: Dict) -> "ExperimentResult":
        """Reconstruct from dictionary."""
        result = ExperimentResult(ExperimentConfig(**data["config"]))
        result.task_matrix.R = np.array(data["task_matrix"], dtype=np.float32)
        result.metrics = data["metrics"]
        result.training_history = data["training_history"]
        return result


# ============================================================================
# EXPERIMENT RUNNER WITH CACHING
# ============================================================================


class ExperimentRunner:
    """
    Orchestrates experiment execution with intelligent caching.

    This class encapsulates the entire experiment workflow:
    - Load and preprocess data
    - Build models
    - Train on task sequences
    - Evaluate and record results
    - Cache results for fast retrieval

    Caching Strategy:
        The runner implements a two-level caching system:

        Level 1: Permuted MNIST Data
            - Files: results/cache/permuted_mnist/task_XX.pkl
            - Cached after first load
            - Reused across all experiments
            - Size: 2.19 GB total (one-time cost)

        Level 2: Experiment Results
            - Files: results/cache/experiments/result_<hash>.json
            - Unique hash per ExperimentConfig
            - Cached after first run
            - Includes: task matrix, metrics, training history

        Benefits:
            - Skip 30 seconds of data loading on subsequent runs
            - Skip 5-10 minutes of experiment execution if re-running same config
            - Enable iteration on visualization without re-running experiments
            - Support resuming interrupted experiment suites

    Attributes:
        seed (int): Random seed for reproducibility
        force_regenerate (bool): If True, ignore caches and re-run everything
        tasks_data (Dict): Loaded permuted MNIST data (cached in memory)

    Key Methods:
        run_experiment(config): Execute single experiment with caching
        run_experiment_suite(configs): Execute multiple experiments
        get_cache_path(config): Get filesystem path for caching
        load_result_from_cache(): Retrieve cached result
        save_result_to_cache(): Store result to disk

    Protocol for run_experiment():
        1. Check cache for results with same config hash
        2. If found: load and return (skip steps 3-8)
        3. Load task data (permuted MNIST)
        4. Create MLP model with specified config
        5. Train on Task 0 for 50 epochs
        6. For Tasks 1-9:
           a. Evaluate on all tasks (record task matrix row)
           b. Train for 20 epochs
        7. Compute all metrics (ACC, BWT, FWT, TBWT, CBWT)
        8. Save to cache and return result

    Example Usage:
        >>> runner = ExperimentRunner(seed=SEED_VALUE, force_regenerate=False)
        >>> config = ExperimentConfig(depth=2, loss_type="nll", ...)
        >>> result = runner.run_experiment(config, verbose=0)
        >>> print(f"ACC: {result.metrics['acc']:.4f}")
        >>> print(f"BWT: {result.metrics['bwt']:.4f}")

    Performance Characteristics:
        First run (clean cache):
        - Data loading: ~30 seconds (MNIST download + permutation)
        - Experiment 1: ~8 minutes (230 total epochs)
        - Experiments 2-42: ~6-8 minutes each
        - Total: 3-5 hours for complete suite

        Subsequent runs (cache hits):
        - Data loading: ~1 second (load from pickle)
        - Per-experiment: <1 second (load JSON)
        - Total: ~30 seconds for complete suite
    """

    def __init__(self, seed: int = SEED_VALUE, force_regenerate: bool = False):
        """
        Initialize runner.

        Args:
            seed: Random seed for reproducibility
            force_regenerate: If True, ignore cache and re-run all experiments
        """
        self.seed = seed
        self.force_regenerate = force_regenerate
        self.tasks_data = None

        logger.info(
            f"ExperimentRunner initialized: "
            f"seed={seed}, force_regenerate={force_regenerate}"
        )

    def _load_tasks(self) -> None:
        """Load all task data (cached after first call)."""
        if self.tasks_data is None:
            logger.info("Loading permuted MNIST data...")
            self.tasks_data = load_all_tasks(self.seed)
            logger.info("Task data loaded successfully")

    def get_cache_path(self, config: ExperimentConfig) -> Path:
        """Get cache file path for a configuration."""
        config_hash = config.get_config_hash()
        return EXPERIMENT_CACHE / f"result_{config_hash}.json"

    def load_result_from_cache(
        self, config: ExperimentConfig
    ) -> Optional[ExperimentResult]:
        """
        Load experiment result from cache if available.

        Args:
            config: Experiment configuration

        Returns:
            ExperimentResult if found, None otherwise
        """
        cache_path = self.get_cache_path(config)

        if cache_path.exists() and not self.force_regenerate:
            try:
                with open(cache_path, "r") as f:
                    data = json.load(f)
                result = ExperimentResult.from_dict(data)
                logger.info(
                    f"Loaded cached result: {config.experiment_name} "
                    f"({cache_path.name})"
                )
                return result
            except Exception as e:
                logger.warning(f"Failed to load cache {cache_path}: {e}")
                return None

        return None

    def save_result_to_cache(self, result: ExperimentResult) -> None:
        """
        Save experiment result to cache.

        Args:
            result: ExperimentResult to save
        """
        cache_path = self.get_cache_path(result.config)

        try:
            with open(cache_path, "w") as f:
                json.dump(result.to_dict(), f, indent=2)
            logger.debug(f"Saved result to cache: {cache_path.name}")
        except Exception as e:
            logger.error(f"Failed to save cache {cache_path}: {e}")

    def run_experiment(
        self, config: ExperimentConfig, verbose: int = 0
    ) -> ExperimentResult:
        """
        Run a single experiment with the given configuration.

        Protocol:
        1. Load task data
        2. Check cache (skip if hit)
        3. Create MLP model
        4. Train on Task 1 for 50 epochs
        5. For each of Tasks 2-9:
           - Evaluate on all tasks (record task matrix row)
           - Train for 20 epochs
        6. Final evaluation on all tasks
        7. Compute metrics
        8. Save to cache

        Args:
            config: ExperimentConfig with all hyperparameters
            verbose: TensorFlow verbosity (0=silent, 1=progress)

        Returns:
            ExperimentResult with task matrix, metrics, history
        """

        # Check cache first - if results already exist for this config, load and return immediately
        # This saves 5-10 minutes per experiment on subsequent runs
        cached = self.load_result_from_cache(config)
        if cached is not None:
            return cached

        logger.info(f"Running experiment: {config.experiment_name}")
        logger.info(f"  Config: {config.to_dict()}")

        # Load task data (permuted MNIST). This is cached after first run,
        # so all 42 experiments reuse the same data (no redundant processing)
        self._load_tasks()

        # Initialize model with config parameters. The model will be trained
        # sequentially on tasks 0-9, with weights persisting between tasks
        # (to simulate continual learning scenario)
        model = MLPClassifier(
            depth=config.depth,
            dropout_rate=config.dropout_rate,
            loss_type=config.loss_type,
            optimizer_type=config.optimizer,
            seed=self.seed,
            verbose=verbose,
        )

        # Initialize result container to record task matrix R[i,j]
        result = ExperimentResult(config)

        # ====== TASK 0: Extended Training (50 epochs) ======
        # Task 0 (standard MNIST with identity permutation) gets more epochs
        # to establish a strong baseline for evaluating forgetting
        logger.info("Training Task 0 (50 epochs)...")
        X_train_0, y_train_0 = self.tasks_data[0][0]  # Training data for Task 0
        X_test_0, y_test_0 = self.tasks_data[0][1]  # Test data for Task 0

        # Train with validation set to monitor generalization
        model.fit(
            X_train_0,
            y_train_0,
            X_val=X_test_0,
            y_val=y_test_0,
            epochs=EPOCHS_TASK_1,
            batch_size=BATCH_SIZE,
            verbose=verbose,
        )

        # Record R[0,:] - accuracy on all tasks immediately after training task 0
        # Should be high on task 0, low on other tasks (haven't seen them yet)
        accuracies_task_0 = self._evaluate_on_all_tasks(model, self.tasks_data)
        result.task_matrix.record_accuracies(0, accuracies_task_0)
        result.training_history.append(
            {
                "task": 0,
                "epoch": EPOCHS_TASK_1,
                "accuracies": accuracies_task_0.tolist(),
            }
        )

        # ====== TASKS 1-9: Sequential Learning (20 epochs each) ======
        # The critical part: train on subsequent tasks and measure forgetting
        for task_id in range(1, NUM_TASKS):
            logger.info(f"Training Task {task_id} ({EPOCHS_PER_TASK} epochs)...")

            # Load task data (different permutation of pixels)
            X_train, y_train = self.tasks_data[task_id][0]
            X_test, y_test = self.tasks_data[task_id][1]

            # Train on new task (weights updated, may overwrite old knowledge)
            model.fit(
                X_train,
                y_train,
                X_val=X_test,
                y_val=y_test,
                epochs=EPOCHS_PER_TASK,
                batch_size=BATCH_SIZE,
                verbose=verbose,
            )

            # Record R[task_id,:] - accuracy on ALL tasks after training task_id
            # Shows degradation on previous tasks (catastrophic forgetting)
            accuracies_after = self._evaluate_on_all_tasks(model, self.tasks_data)
            result.task_matrix.record_accuracies(task_id, accuracies_after)
            result.training_history.append(
                {
                    "task": task_id,
                    "epoch": EPOCHS_PER_TASK,
                    "accuracies": accuracies_after.tolist(),
                }
            )

        # Compute all metrics from final task matrix R[9,:]
        # ACC: average performance across all tasks
        # BWT: average degradation on previous tasks (forgetting measure)
        # FWT: average benefit to future tasks from prior learning
        result.metrics = compute_all_metrics(result.task_matrix.R)

        logger.info(f"Experiment complete: {config.experiment_name}")
        logger.info(f"  Metrics: {result.metrics}")

        # Save to cache for future runs
        self.save_result_to_cache(result)

        return result

    def _evaluate_on_all_tasks(
        self, model: MLPClassifier, tasks_data: Dict
    ) -> np.ndarray:
        """
        Evaluate model on all tasks, return accuracy vector.

        Args:
            model: Trained MLPClassifier
            tasks_data: Dictionary of task data from load_all_tasks()

        Returns:
            np.ndarray of shape (NUM_TASKS,) with accuracy on each task
        """
        accuracies = np.zeros(NUM_TASKS, dtype=np.float32)

        for task_id in range(NUM_TASKS):
            X_test, y_test = tasks_data[task_id][1]
            _, acc = model.evaluate(X_test, y_test)
            accuracies[task_id] = acc

        return accuracies

    def run_experiment_suite(
        self, configs: list, verbose: int = 0
    ) -> Dict[str, ExperimentResult]:
        """
        Run multiple experiments.

        Args:
            configs: List of ExperimentConfig objects
            verbose: TensorFlow verbosity

        Returns:
            Dictionary mapping experiment name → ExperimentResult
        """
        results = {}
        total = len(configs)

        for i, config in enumerate(configs):
            logger.info(f"[{i + 1}/{total}] Running {config.experiment_name}...")
            result = self.run_experiment(config, verbose=verbose)
            results[config.experiment_name] = result

        logger.info(f"Experiment suite complete: {total} experiments")
        return results
