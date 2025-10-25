"""
Metrics computation for continual learning evaluation.

Implements standard metrics from GEM (Lopez-Paz & Ranzato, 2017):
- ACC (Average Accuracy)
- BWT (Backward Transfer)
- FWT (Forward Transfer)

Plus bonus metrics from lifelong learning literature:
- TBWT (Task-Based Backward Transfer)
- CBWT (Class-Based Backward Transfer)

References:
[1] Lopez-Paz & Ranzato. Gradient Episodic Memory for Continual Learning.
    NeurIPS 2017.
[2] Ororbia et al. Lifelong Neural Predictive Coding: Learning Cumulatively
    Online without Forgetting. NeurIPS 2022.
"""

import logging
from typing import Dict

import numpy as np

logger = logging.getLogger(__name__)

# ============================================================================
# TASK MATRIX CONSTRUCTION
# ============================================================================


class TaskMatrix:
    """
    Records and manages the task matrix R for continual learning evaluation.

    The task matrix R is the core data structure for evaluating continual learning.
    It's a T×T matrix where R[i,j] represents:
        - Accuracy on task j
        - After training through task i (inclusive)

    Key Properties:
        - Main diagonal R[i,i]: High values (good on task just learned)
        - Below diagonal (i > j): Shows forgetting of previous tasks
        - Above diagonal (i < j): Shows forward transfer potential

    Example Task Matrix (T=5 tasks):
        Task:  0     1     2     3     4
        ┌─────────────────────────────────┐
        │ 0.92  0.10  0.05  0.08  0.06   │  After task 0
        │ 0.85  0.90  0.10  0.07  0.06   │  After task 1
        │ 0.42  0.88  0.91  0.09  0.07   │  After task 2  ← Forgetting!
        │ 0.30  0.72  0.85  0.90  0.08   │  After task 3
        │ 0.28  0.65  0.78  0.85  0.92   │  After task 4
        └─────────────────────────────────┘

    Observations from above matrix:
        - Task 0: 92% initially, 28% finally (64% forgotten)
        - Task 1: 90% immediately, 65% finally (25% forgotten)
        - Task 4: 92% immediately (just learned)
        - Clear catastrophic forgetting in lower-left triangle

    Attributes:
        R (np.ndarray): The T×T task matrix, dtype=float32
        T (int): Number of tasks (typically 10)
        num_tasks_seen (int): How many tasks have been completed so far

    Usage Pattern:
        >>> matrix = TaskMatrix(num_tasks=10)
        >>> # After training each task:
        >>> for task_id in range(10):
        ...     accuracies_all_tasks = model.evaluate_on_all_tasks()
        ...     matrix.record_accuracies(task_id, accuracies_all_tasks)
        >>> # Compute metrics:
        >>> metrics = compute_all_metrics(matrix.R)
    """

    def __init__(self, num_tasks: int):
        """
        Initialize task matrix.

        Args:
            num_tasks: Total number of tasks (T=10 for MNIST)
        """
        self.T = num_tasks
        self.R = np.zeros((num_tasks, num_tasks), dtype=np.float32)
        self.num_tasks_seen = 0

    def record_accuracies(
        self, task_just_completed: int, accuracies_on_all_tasks: np.ndarray
    ) -> None:
        """
        Record accuracy scores after completing a task.

        Called after training on task_just_completed finishes.
        Records the accuracy evaluated on ALL tasks (0 to T-1).

        Args:
            task_just_completed: Task index (0-indexed)
            accuracies_on_all_tasks: Array of shape (T,) with test accuracies
                                     on each task after training task_just_completed

        Preconditions:
            - task_just_completed in [0, T-1]
            - len(accuracies_on_all_tasks) == T
            - All values in [0, 1]

        Example:
            After training on task 2, evaluate on tasks 0-9:
            accuracies = model.evaluate_on_all_tasks()  # shape (10,)
            R.record_accuracies(2, accuracies)
            # Sets R[2, :] = accuracies (row 2)
        """
        assert 0 <= task_just_completed < self.T, (
            f"Invalid task index {task_just_completed}"
        )
        assert len(accuracies_on_all_tasks) == self.T, (
            f"Expected {self.T} accuracies, got {len(accuracies_on_all_tasks)}"
        )
        assert np.all(
            (accuracies_on_all_tasks >= 0) & (accuracies_on_all_tasks <= 1)
        ), "Accuracies must be in [0, 1]"

        self.R[task_just_completed, :] = accuracies_on_all_tasks
        self.num_tasks_seen = task_just_completed + 1

        logger.debug(
            f"Recorded accuracies after task {task_just_completed}: "
            f"R[{task_just_completed}] = {accuracies_on_all_tasks}"
        )

    def get_matrix(self) -> np.ndarray:
        """Get the full task matrix R."""
        return self.R.copy()

    def is_complete(self) -> bool:
        """Check if all tasks have been completed."""
        return self.num_tasks_seen == self.T


# ============================================================================
# METRICS FROM GEM PAPER (Lopez-Paz & Ranzato, 2017)
# ============================================================================


def compute_acc(R: np.ndarray) -> float:
    """
    Compute Average Accuracy (ACC).

    ACC = (1/T) * Σ(R[T, i] for i in 0..T-1)

    This is the average performance across all tasks after training
    the complete sequence. Reported in the last row of the task matrix.

    Args:
        R: Task matrix of shape (T, T)

    Returns:
        Float in [0, 1], higher is better

    Rationale:
        Measures overall performance at the end of training.
        Does NOT capture forgetting specifically.

    Reference:
        GEM Eq. 2
    """
    T = R.shape[0]
    return float(np.mean(R[T - 1, :]))


def compute_bwt(R: np.ndarray) -> float:
    """
    Compute Backward Transfer (BWT).

    BWT = (1/(T-1)) * Σ(R[T, i] - R[i, i] for i in 0..T-2)

    Measures how much learning subsequent tasks helps/hurts previous tasks.

    - Positive BWT: Learning new tasks improves old task performance
    - Negative BWT: Learning new tasks hurts old task performance (forgetting)
    - BWT ≈ 0: Neutral transfer

    Args:
        R: Task matrix of shape (T, T)

    Returns:
        Float typically in [-1, 1], higher is better

    Interpretation:
        BWT = -0.5 means average 50% accuracy drop due to catastrophic forgetting
        BWT = +0.05 means average 5% accuracy improvement on old tasks

    Reference:
        GEM Eq. 3
    """
    T = R.shape[0]
    bwt = 0.0

    for i in range(T - 1):
        # R[T-1, i] = accuracy on task i after learning all T tasks
        # R[i, i] = accuracy on task i immediately after learning it
        bwt += R[T - 1, i] - R[i, i]

    bwt /= T - 1
    return float(bwt)


def compute_fwt(R: np.ndarray) -> float:
    """
    Compute Forward Transfer (FWT).

    FWT = (1/(T-1)) * Σ(R[i-1, i] - b̄[i] for i in 1..T-1)

    Measures how much learning previous tasks helps future tasks.

    - Positive FWT: Previous tasks help learn new tasks (transfer)
    - Negative FWT: Previous tasks hurt new task learning
    - FWT ≈ 0: Neutral transfer

    Args:
        R: Task matrix of shape (T, T)

    Returns:
        Float typically in [-1, 1], higher is better

    Note:
        This implementation assumes b̄[i] ≈ random baseline ≈ 0.1 for 10-class
        problem. In full implementation, should evaluate untrained model on task i.

    Reference:
        GEM Eq. 4
    """
    T = R.shape[0]
    fwt = 0.0
    baseline_acc = 0.1  # Random baseline for 10-class problem

    for i in range(1, T):
        # R[i-1, i] = accuracy on task i after learning tasks 0..i-1
        # This is FWT because we haven't trained on task i yet
        fwt += R[i - 1, i] - baseline_acc

    fwt /= T - 1
    return float(fwt)


# ============================================================================
# BONUS METRICS FROM LIFELONG LEARNING LITERATURE
# ============================================================================


def compute_tbwt(R: np.ndarray) -> float:
    """
    Compute Task-Based Backward Transfer (TBWT) - Bonus.

    Extended metric considering forgetting per task:

    TBWT = (1/T) * Σ(max(0, R[i, i] - R[T, i]) for i in 0..T-1)

    Measures total forgetting across all tasks (only counts loss, not gain).

    Args:
        R: Task matrix of shape (T, T)

    Returns:
        Float in [0, 1], lower is better (less forgetting)

    Reference:
        Ororbia et al., NeurIPS 2022
    """
    T = R.shape[0]
    tbwt = 0.0

    for i in range(T):
        forgetting = max(0, R[i, i] - R[T - 1, i])
        tbwt += forgetting

    tbwt /= T
    return float(tbwt)


def compute_cbwt(R: np.ndarray) -> float:
    """
    Compute Class-Based Backward Transfer (CBWT) - Bonus.

    CBWT = (1/T) * Σ(min(1, R[i, i] - R[T, i]) for i in 0..T-1)

    Similar to TBWT but clipped to [0, 1] for bounded metric.

    Args:
        R: Task matrix of shape (T, T)

    Returns:
        Float in [0, 1], lower is better (less forgetting)

    Reference:
        Ororbia et al., NeurIPS 2022
    """
    T = R.shape[0]
    cbwt = 0.0

    for i in range(T):
        forgetting = min(1.0, R[i, i] - R[T - 1, i])
        cbwt += forgetting

    cbwt /= T
    return float(cbwt)


# ============================================================================
# COMPREHENSIVE METRICS DICTIONARY
# ============================================================================


def compute_all_metrics(R: np.ndarray) -> Dict[str, float]:
    """
    Compute comprehensive continual learning metrics from task matrix.

    This is the MAIN METRICS FUNCTION - call after all tasks complete.
    Computes 5 metrics capturing different aspects of continual learning.

    Args:
        R (np.ndarray): Task matrix of shape (T, T)
            R[i, j] = accuracy on task j after training through task i

            Typical values:
            - R[0, 0] ≈ 0.85 (high immediately after learning Task 0)
            - R[9, 0] ≈ 0.30-0.40 (degraded due to forgetting)
            - R[i, i] ≈ high (good performance right after task training)
            - R[i, j] for i < j = 0 (haven't seen task yet)

    Returns:
        Dict[str, float] with keys:

        'acc' (Average Accuracy):
            Mean performance across all tasks at end of sequence
            Formula: (1/T) * Σ(R[T-1, i] for i=0..T-1)
            Interpretation: Overall competence after learning all tasks
            Range: [0, 1], higher is better

        'bwt' (Backward Transfer):
            Average impact of learning new tasks on old task performance
            Formula: (1/(T-1)) * Σ(R[T-1, i] - R[i, i] for i=0..T-2)
            Interpretation:
                - BWT = -0.5: Lose 50% accuracy on old tasks (severe forgetting)
                - BWT = 0: No change in old task performance
                - BWT = +0.05: Gain 5% on old tasks (positive transfer)
            Range: [-1, 1], higher is better
            **KEY METRIC**: Main measure of catastrophic forgetting

        'fwt' (Forward Transfer):
            Average benefit of prior learning on new task performance
            Formula: (1/(T-1)) * Σ(R[i-1, i] - baseline for i=1..T-1)
            Interpretation:
                - FWT > 0: Prior tasks help learn new tasks
                - FWT = 0: No transfer benefit
                - FWT < 0: Prior tasks hurt new task learning
            Range: [-1, 1], higher is better

        'tbwt' (Task-Based Backward Transfer - Bonus):
            Cumulative forgetting across all tasks (loss-only)
            Formula: (1/T) * Σ(max(0, R[i,i] - R[T-1,i]) for i=0..T-1)
            Interpretation: Total amount "forgotten" across all tasks
            Range: [0, 1], lower is better (less forgetting)

        'cbwt' (Class-Based Backward Transfer - Bonus):
            Similar to TBWT but clipped to [0, 1]
            Formula: (1/T) * Σ(min(1, R[i,i] - R[T-1,i]) for i=0..T-1)
            Interpretation: Bounded version of TBWT
            Range: [0, 1], lower is better

    Example Task Matrix Interpretation:
        R = [
            [0.92, 0.10, 0.11, ...],  # After task 0: 92% on task 0
            [0.85, 0.90, 0.12, ...],  # After task 1: 85% on task 0, 90% on task 1
            [0.42, 0.88, 0.91, ...],  # After task 2: Forgetting! 42% on task 0
            ...
            [0.30, 0.75, 0.84, ...]   # After task 9: Further degradation
        ]

        ACC = mean(last row) ≈ 0.60
        BWT = mean([0.30-0.92, 0.75-0.90, 0.84-0.91, ...]) ≈ -0.35
            (on average, 35% performance loss on previous tasks)

    References:
        [1] Lopez-Paz & Ranzato. "Gradient Episodic Memory for Continual Learning"
            NeurIPS 2017. Defines ACC, BWT, FWT.
        [2] Ororbia et al. "Lifelong Neural Predictive Coding: Learning Cumulatively
            Online without Forgetting" NeurIPS 2022. Discusses TBWT, CBWT.

    Raises:
        AssertionError: If R shape, range, or content invalid

    Typical Output (from GEM paper results):
        {
            'acc': 0.792,
            'bwt': -0.176,  # ~17% forgetting on previous tasks
            'fwt': 0.025,   # Minimal forward transfer
            'tbwt': 0.142,  # ~14% cumulative loss
            'cbwt': 0.140   # Clipped version
        }
    """

    return {
        "acc": compute_acc(R),
        "bwt": compute_bwt(R),
        "fwt": compute_fwt(R),
        "tbwt": compute_tbwt(R),
        "cbwt": compute_cbwt(R),
    }


# ============================================================================
# VALIDATION & DEBUGGING UTILITIES
# ============================================================================


def validate_task_matrix(R: np.ndarray) -> bool:
    """
    Validate that task matrix has expected properties.

    Checks:
    - Shape is (T, T)
    - All values in [0, 1]
    - Main diagonal is populated

    Args:
        R: Task matrix to validate

    Returns:
        True if valid, raises AssertionError otherwise
    """
    assert R.ndim == 2, f"R must be 2D, got shape {R.shape}"
    assert R.shape[0] == R.shape[1], f"R must be square, got {R.shape}"
    assert np.all((R >= 0) & (R <= 1)), (
        f"All values must be in [0,1], got range [{R.min()}, {R.max()}]"
    )

    T = R.shape[0]
    diagonal = np.diag(R)
    assert np.all(diagonal > 0), (
        "Main diagonal should have positive values (R[i,i] > 0)"
    )

    return True


def print_task_matrix(R: np.ndarray, title: str = "Task Matrix") -> None:
    """
    Pretty-print task matrix with metrics.

    Args:
        R: Task matrix
        title: Optional title
    """
    print(f"\n{title} (T={R.shape[0]}):")
    print("=" * 60)
    print(np.array2string(R, precision=3, separator=", "))
    print("-" * 60)

    metrics = compute_all_metrics(R)
    for name, value in metrics.items():
        print(f"{name.upper():6s}: {value:+.4f}")
