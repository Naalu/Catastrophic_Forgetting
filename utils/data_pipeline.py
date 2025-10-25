"""
Data pipeline for permuted MNIST experiments.

Handles:
- Loading MNIST dataset
- Generating deterministic permutations per task
- Caching permuted data to disk
- Validation of data integrity
"""

import logging
import pickle
from typing import Dict, Tuple

import numpy as np
import tensorflow as tf
from config import (
    INPUT_SHAPE,
    MNIST_TEST_SIZE,
    MNIST_TRAIN_SIZE,
    NUM_CLASSES,
    NUM_TASKS,
    PERMUTED_MNIST_CACHE,
    SEED_VALUE,
)

logger = logging.getLogger(__name__)

# ============================================================================
# PERMUTATION MANAGEMENT
# ============================================================================


def generate_permutation(task_id: int, seed: int) -> np.ndarray:
    """
    Generate deterministic pixel permutation for a task.

    Args:
        task_id: Task number (0-indexed, 0-9)
        seed: Random seed for reproducibility

    Returns:
        np.ndarray of shape (784,) containing permutation indices

    Invariants:
        - Same (task_id, seed) always produces identical permutation
        - Each permutation is a shuffle of range(784)
        - Permutation is identity for task 0 (baseline)

    Rationale:
        Task 0 uses identity permutation (standard MNIST) to establish
        baseline performance. Subsequent tasks scramble pixel positions
        to create distribution shift while keeping data identical.
    """
    rng = np.random.RandomState(seed + task_id)  # Task-specific seed

    if task_id == 0:
        # Task 0: identity permutation (no scrambling)
        return np.arange(INPUT_SHAPE[0])
    else:
        # Tasks 1-9: random permutations
        return rng.permutation(INPUT_SHAPE[0])


def apply_permutation(X: np.ndarray, permutation: np.ndarray) -> np.ndarray:
    """
    Apply pixel permutation to image batch.

    Args:
        X: Array of shape (N, 784) or (784,)
        permutation: Permutation indices from generate_permutation()

    Returns:
        Permuted array with same shape as X

    Validates:
        - X is 1D or 2D
        - X has 784 features
        - permutation length matches 784
    """
    assert len(permutation) == INPUT_SHAPE[0], (
        f"Permutation length {len(permutation)} != input shape {INPUT_SHAPE[0]}"
    )

    if X.ndim == 1:
        return X[permutation]
    else:
        assert X.shape[1] == INPUT_SHAPE[0], (
            f"X features {X.shape[1]} != input shape {INPUT_SHAPE[0]}"
        )
        return X[:, permutation]


# ============================================================================
# MNIST LOADING & PREPARATION
# ============================================================================


def load_mnist_raw() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load raw MNIST dataset from TensorFlow.

    Returns:
        (X_train, y_train, X_test, y_test) with:
        - X: shape (N, 28, 28), values in [0, 255], dtype uint8
        - y: shape (N,), values in [0, 9], dtype uint8

    Raises:
        RuntimeError: If download or loading fails
    """
    try:
        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

        assert X_train.shape == (MNIST_TRAIN_SIZE, 28, 28), (
            f"Train shape mismatch: {X_train.shape}"
        )
        assert X_test.shape == (MNIST_TEST_SIZE, 28, 28), (
            f"Test shape mismatch: {X_test.shape}"
        )
        assert X_train.dtype == np.uint8
        assert y_train.min() == 0 and y_train.max() == 9

        logger.info(f"Loaded MNIST: train {X_train.shape}, test {X_test.shape}")
        return X_train, y_train, X_test, y_test

    except Exception as e:
        logger.error(f"Failed to load MNIST: {e}")
        raise RuntimeError("MNIST loading failed") from e


def preprocess_mnist(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess MNIST data for training.

    Args:
        X: Shape (N, 28, 28), dtype uint8
        y: Shape (N,), dtype uint8

    Returns:
        (X_processed, y_processed) with:
        - X: Shape (N, 784), dtype float32, normalized to [0, 1]
        - y: Shape (N, 10), dtype float32, one-hot encoded

    Rationale:
        - Flattening: Convert 2D images to feature vectors
        - Normalization: [0,255] → [0,1] for numerical stability
        - One-hot: Standard for categorical cross-entropy loss
    """
    # Flatten
    X_flat = X.reshape(X.shape[0], -1).astype(np.float32)

    # Normalize
    X_normalized = X_flat / 255.0

    # One-hot encode labels
    y_onehot = tf.keras.utils.to_categorical(y, NUM_CLASSES).astype(np.float32)

    assert X_normalized.shape == (X.shape[0], 784)
    assert X_normalized.min() >= 0.0 and X_normalized.max() <= 1.0
    assert y_onehot.shape == (y.shape[0], NUM_CLASSES)

    logger.debug(f"Preprocessed: X {X_normalized.shape}, y {y_onehot.shape}")
    return X_normalized, y_onehot


# ============================================================================
# PERMUTED MNIST DATASET GENERATION & CACHING
# ============================================================================


def load_permuted_task_data(
    task_id: int, seed: int = SEED_VALUE, force_regenerate: bool = False
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Load preprocessed, permuted MNIST data for a specific task.

    This is the PRIMARY DATA LOADING FUNCTION for experiments.
    It abstracts away caching, permutation generation, and preprocessing.

    Args:
        task_id (int): Task identifier in [0, 9]
            - Task 0: Identity permutation (standard MNIST)
            - Tasks 1-9: Random permutations of pixel positions

        seed (int): Random seed for permutation generation
            - Default: get_seed_value("Karl") ≈ 3.2M
            - Identical seed + task_id always produces same permutation

        force_regenerate (bool): If True, ignore cache and regenerate
            - Use when debugging permutation generation
            - Default: False (use cache for speed)

    Returns:
        Tuple of form:
        (
            (X_train, y_train),    # Training set
            (X_test, y_test)       # Test set
        )

        Where:
        - X_train: shape (60000, 784), dtype float32, values in [0, 1]
        - y_train: shape (60000, 10), dtype float32, one-hot encoded
        - X_test: shape (10000, 784), dtype float32, values in [0, 1]
        - y_test: shape (10000, 10), dtype float32, one-hot encoded

    Caching Behavior:
        First call (or force_regenerate=True):
            1. Downloads MNIST from TensorFlow (if not cached)
            2. Flattens 28×28 images to 784 features
            3. Normalizes pixel values to [0, 1]
            4. Generates permutation deterministically
            5. Applies permutation to training + test data
            6. One-hot encodes labels
            7. Saves to ~/results/cache/permuted_mnist/task_XX.pkl

        Subsequent calls (cache hit):
            - Loads from pickle (≈0.2 seconds for all 10 tasks)

    Performance:
        First run: ~30 seconds (MNIST download + processing)
        Cached runs: ~1 second (load 10 files from disk)

    Raises:
        ValueError: If task_id not in [0, 9]
        RuntimeError: If MNIST download fails
        IOError: If cache file corrupted

    Examples:
        >>> # Load Task 0 (standard MNIST)
        >>> (X_train, y_train), (X_test, y_test) = load_permuted_task_data(0)
        >>> X_train.shape
        (60000, 784)
        >>> y_train.shape
        (60000, 10)

        >>> # Load Task 5 with custom seed
        >>> custom_seed = 12345
        >>> (X_train, y_train), _ = load_permuted_task_data(5, seed=custom_seed)

        >>> # Force regeneration (useful for debugging)
        >>> load_permuted_task_data(0, force_regenerate=True)

    Notes:
        - All tasks use identical labels (only pixels permuted)
        - Permutation is deterministic: same (task_id, seed) → same permutation
        - Permutation is fixed per task (not randomized per epoch)
        - One-hot encoding uses standard TensorFlow utility
    """

    assert 0 <= task_id < NUM_TASKS, f"task_id {task_id} not in [0, {NUM_TASKS - 1}]"

    # Generate cache filename
    cache_file = PERMUTED_MNIST_CACHE / f"task_{task_id:02d}.pkl"

    # Check cache
    if cache_file.exists() and not force_regenerate:
        try:
            with open(cache_file, "rb") as f:
                data = pickle.load(f)
            logger.info(f"Loaded task {task_id} from cache: {cache_file}")
            return data
        except Exception as e:
            logger.warning(f"Cache load failed for task {task_id}: {e}, regenerating")

    # Generate from scratch
    try:
        logger.info(f"Generating permuted MNIST for task {task_id}...")

        # Load raw MNIST
        X_train_raw, y_train_raw, X_test_raw, y_test_raw = load_mnist_raw()

        # Preprocess
        X_train, y_train = preprocess_mnist(X_train_raw, y_train_raw)
        X_test, y_test = preprocess_mnist(X_test_raw, y_test_raw)

        # Generate permutation
        perm = generate_permutation(task_id, seed)

        # Apply permutation
        X_train_perm = apply_permutation(X_train, perm)
        X_test_perm = apply_permutation(X_test, perm)

        data = ((X_train_perm, y_train), (X_test_perm, y_test))

        # Save to cache
        with open(cache_file, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"Saved task {task_id} to cache: {cache_file}")

        return data

    except Exception as e:
        logger.error(f"Failed to generate task {task_id} data: {e}")
        raise RuntimeError(f"Data generation failed for task {task_id}") from e


def load_all_tasks(
    seed: int = SEED_VALUE, force_regenerate: bool = False
) -> Dict[int, Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]]:
    """
    Load all 10 tasks' permuted MNIST data.

    Args:
        seed: Random seed for reproducibility
        force_regenerate: Regenerate all caches

    Returns:
        Dictionary mapping task_id → ((X_train, y_train), (X_test, y_test))

    Performance:
        - First call: ~30 seconds (download MNIST, generate, cache)
        - Subsequent calls: ~1 second (load from cache)
    """
    tasks = {}
    for task_id in range(NUM_TASKS):
        tasks[task_id] = load_permuted_task_data(task_id, seed, force_regenerate)
        logger.debug(f"Loaded task {task_id}")

    logger.info(f"All {NUM_TASKS} tasks loaded successfully")
    return tasks


# ============================================================================
# BATCH GENERATOR FOR SEQUENTIAL TRAINING
# ============================================================================


class SequentialTaskDataGenerator:
    """
    Generator for sequential task learning protocol.

    Yields batches from a single task for one epoch, then moves to next task.
    Implements the training protocol: 50 epochs task 1, then 20 epochs each.

    Attributes:
        tasks: Dict of task data from load_all_tasks()
        batch_size: Batch size for training
        current_task: Currently active task (0-indexed)
        current_epoch: Epoch within current task
        is_training: Whether in training mode (affects epoch counting)
    """

    def __init__(
        self,
        tasks: Dict[
            int, Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]
        ],
        batch_size: int = 32,
    ):
        """
        Initialize generator.

        Args:
            tasks: Output from load_all_tasks()
            batch_size: Batch size for training
        """
        self.tasks = tasks
        self.batch_size = batch_size
        self.current_task = 0
        self.current_epoch = 0
        self.is_training = True

    def get_task_data(self, task_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get data for a task (train or test).

        Args:
            task_id: Task number

        Returns:
            (X, y) tuple
        """
        if self.is_training:
            return self.tasks[task_id][0]  # (X_train, y_train)
        else:
            return self.tasks[task_id][1]  # (X_test, y_test)

    def get_batches_for_task(self, task_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get batches for one epoch of a task.

        Args:
            task_id: Task number

        Yields:
            (X_batch, y_batch) tuples
        """
        X, y = self.get_task_data(task_id)
        num_samples = X.shape[0]
        num_batches = (num_samples + self.batch_size - 1) // self.batch_size

        # Shuffle for training, keep order for testing
        if self.is_training:
            indices = np.random.permutation(num_samples)
        else:
            indices = np.arange(num_samples)

        for i in range(num_batches):
            start_idx = i * self.batch_size
            end_idx = min(start_idx + self.batch_size, num_samples)
            batch_indices = indices[start_idx:end_idx]

            yield X[batch_indices], y[batch_indices]
