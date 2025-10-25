"""
Multi-layer Perceptron architecture for continual learning experiments.

Implements flexible MLP with configurable:
- Network depth (2, 3, or 4 hidden layers)
- Hidden layer size (256 units)
- Dropout rates
- Loss functions (NLL, L1, L2, L1+L2)
- Optimizers (SGD, Adam, RMSprop)

Uses TensorFlow 2.x with Metal GPU acceleration for Apple Silicon.
"""

import logging
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf
from config import (
    HIDDEN_UNITS,
    INPUT_SHAPE,
    NUM_CLASSES,
    OPTIMIZER_CONFIG,
    REGULARIZATION_STRENGTH,
    LossType,
    OptimizerType,
)

logger = logging.getLogger(__name__)

# ============================================================================
# LOSS FUNCTION BUILDERS
# ============================================================================


def build_loss_function(loss_type: str, seed: int = 42) -> Callable:
    """
    Build loss function based on type.

    Args:
        loss_type: One of "nll", "l1", "l2", "l1_l2"
        seed: Random seed for reproducibility

    Returns:
        Loss function callable (y_true, y_pred) → scalar loss

    Implements:
        - NLL: Standard categorical cross-entropy (baseline)
        - L1: CE + L1 regularization on weights
        - L2: CE + L2 regularization on weights
        - L1+L2: CE + combined regularization

    Note:
        Regularization terms are applied as empirical regularization
        (added to loss), not as optimizer weight decay.
    """

    if loss_type == LossType.NLL.value:
        return tf.keras.losses.CategoricalCrossentropy(from_logits=False)

    elif loss_type == LossType.L1.value:
        reg_strength = REGULARIZATION_STRENGTH[LossType.L1]

        def l1_loss(y_true, y_pred):
            ce = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
            return tf.reduce_mean(ce) + reg_strength * tf.reduce_sum(tf.abs(y_pred))

        return l1_loss

    elif loss_type == LossType.L2.value:
        reg_strength = REGULARIZATION_STRENGTH[LossType.L2]

        def l2_loss(y_true, y_pred):
            ce = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
            return tf.reduce_mean(ce) + reg_strength * tf.reduce_sum(tf.square(y_pred))

        return l2_loss

    elif loss_type == LossType.L1_L2.value:
        reg_strength = REGULARIZATION_STRENGTH[LossType.L1_L2]

        def l1_l2_loss(y_true, y_pred):
            ce = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
            l1_term = reg_strength * tf.reduce_sum(tf.abs(y_pred))
            l2_term = reg_strength * tf.reduce_sum(tf.square(y_pred))
            return tf.reduce_mean(ce) + l1_term + l2_term

        return l1_l2_loss

    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def build_optimizer(
    optimizer_type: str, seed: int = 42
) -> tf.keras.optimizers.Optimizer:
    """
    Build optimizer with configured hyperparameters.

    Args:
        optimizer_type: One of "sgd", "adam", "rmsprop"
        seed: Random seed for reproducibility

    Returns:
        TensorFlow optimizer instance

    Implements:
        - SGD: Stochastic Gradient Descent with momentum
        - Adam: Adaptive Moment Estimation
        - RMSprop: Root Mean Square Propagation

    Each optimizer has tuned hyperparameters from OPTIMIZER_CONFIG.
    """

    if optimizer_type == OptimizerType.SGD.value:  # "sgd"
        config = OPTIMIZER_CONFIG[OptimizerType.SGD]
        return tf.keras.optimizers.SGD(
            learning_rate=config["learning_rate"],
            momentum=config["momentum"],
            seed=seed,
        )

    elif optimizer_type == OptimizerType.ADAM.value:  # "adam"
        config = OPTIMIZER_CONFIG[OptimizerType.ADAM]
        return tf.keras.optimizers.Adam(
            learning_rate=config["learning_rate"],
            beta_1=config["beta_1"],
            beta_2=config["beta_2"],
            seed=seed,
        )

    elif optimizer_type == OptimizerType.RMSPROP.value:  # "rmsprop"
        config = OPTIMIZER_CONFIG[OptimizerType.RMSPROP]
        return tf.keras.optimizers.RMSprop(
            learning_rate=config["learning_rate"],
            rho=config["rho"],
            seed=seed,
        )

    else:
        raise ValueError(
            f"Unknown optimizer: {optimizer_type}. "
            f"Must be one of: {[o.value for o in OptimizerType]}"
        )


# ============================================================================
# MLP CLASSIFIER
# ============================================================================


class MLPClassifier:
    """
    Multi-layer Perceptron for continual learning on permuted MNIST.

    This class wraps TensorFlow Keras to provide a clean interface for:
    - Building MLPs with configurable depth and dropout
    - Training on sequential tasks
    - Evaluating on multiple tasks
    - Extracting weights for analysis

    Architecture Design:
        The network follows a standard feedforward architecture:

        Input (784 pixels)
             ↓
        Dense(256) + ReLU
             ↓
        [Dropout]  (if dropout_rate > 0)
             ↓
        Dense(256) + ReLU
             ↓
        [Dropout]
             ↓
        ... (repeated depth times)
             ↓
        Dense(10) + Softmax  (output layer, one per class)
             ↓
        Predictions (class probabilities)

    Key Design Choices:
        1. ReLU activation: Standard for deep networks, numerically stable
        2. Softmax output: Produces valid probability distribution
        3. Dropout as regularization: Prevents overfitting to individual tasks
        4. Fixed hidden layer size (256): Provides capacity for complex tasks
        5. Glorot uniform weight initialization: Balances gradient flow

    Attributes:
        depth (int): Number of hidden layers (2, 3, or 4)
        hidden_units (int): Units per hidden layer (always 256)
        dropout_rate (float): Dropout probability (0.0, 0.2, 0.5)
        loss_type (str): Loss function type (nll, l1, l2, l1_l2)
        optimizer_type (str): Optimizer (sgd, adam, rmsprop)
        model (tf.keras.Sequential): The underlying TensorFlow model
        history (Dict): Training history (losses, accuracies per epoch)
        seed (int): Random seed for reproducibility

    Example Usage:
        >>> model = MLPClassifier(
        ...     depth=3,
        ...     dropout_rate=0.2,
        ...     loss_type="nll",
        ...     optimizer_type="adam",
        ...     seed=12345
        ... )
        >>> # Train on Task 1
        >>> model.fit(X_train, y_train, epochs=50, batch_size=32)
        >>> # Evaluate on Task 2
        >>> loss, acc = model.evaluate(X_test, y_test)
        >>> # Get predictions
        >>> predictions = model.predict(X_test)

    Internal Methods:
        _build_model(): Constructs the Sequential model with specified architecture
        fit(): Training wrapper around model.fit()
        evaluate(): Evaluation wrapper around model.evaluate()
        predict(): Prediction wrapper

    Important Notes:
        - TensorFlow seed set in __init__ for reproducibility
        - Model compiled immediately after building
        - History accumulated across multiple fit() calls (sequential learning)
        - Weights persist between tasks (intentional for continual learning study)
    """

    def __init__(
        self,
        depth: int,
        dropout_rate: float,
        loss_type: str,
        optimizer_type: str,
        seed: int = 42,
        verbose: int = 0,
    ):
        """
        Initialize MLP classifier.

        Args:
            depth: Number of hidden layers (2, 3, or 4)
            dropout_rate: Dropout probability (0.0, 0.2, 0.5)
            loss_type: Loss function type
            optimizer_type: Optimizer type
            seed: Random seed for reproducibility
            verbose: Logging verbosity (0=silent, 1=progress, 2=detailed)

        Validates:
            - depth in {2, 3, 4}
            - dropout_rate in {0.0, 0.2, 0.5}
        """

        assert depth in [2, 3, 4], f"Depth must be 2, 3, or 4, got {depth}"
        assert dropout_rate in [0.0, 0.2, 0.5], (
            f"Dropout must be 0.0, 0.2, or 0.5, got {dropout_rate}"
        )

        self.depth = depth
        self.hidden_units = HIDDEN_UNITS
        self.dropout_rate = dropout_rate
        self.loss_type = loss_type
        self.optimizer_type = optimizer_type
        self.seed = seed
        self.verbose = verbose

        # Set TensorFlow seed for reproducibility
        tf.random.set_seed(seed)
        np.random.seed(seed)

        # Build model
        self.model = self._build_model()

        # Compile model
        loss_fn = build_loss_function(loss_type, seed)
        optimizer = build_optimizer(optimizer_type, seed)

        self.model.compile(loss=loss_fn, optimizer=optimizer, metrics=["accuracy"])

        # Training history
        self.history = {
            "losses": [],
            "accuracies": [],
            "val_losses": [],
            "val_accuracies": [],
        }

        logger.info(
            f"Initialized MLPClassifier: "
            f"depth={depth}, dropout={dropout_rate}, "
            f"loss={loss_type}, optimizer={optimizer_type}"
        )

    def _build_model(self) -> tf.keras.Sequential:
        """
        Build the MLP architecture.

        Returns:
            TensorFlow Sequential model

        Architecture pattern:
            Input (784)
            ↓
            [Dense(256) + ReLU + Dropout] ×depth
            ↓
            Dense(10) + Softmax (output)
        """

        model = tf.keras.Sequential()

        # Input layer (implicit)
        model.add(tf.keras.layers.Input(shape=INPUT_SHAPE))

        # Hidden layers
        for i in range(self.depth):
            model.add(
                tf.keras.layers.Dense(
                    self.hidden_units,
                    activation="relu",
                    kernel_initializer="glorot_uniform",
                    bias_initializer="zeros",
                    name=f"dense_{i}",
                )
            )

            if self.dropout_rate > 0:
                model.add(
                    tf.keras.layers.Dropout(
                        self.dropout_rate, seed=self.seed + i, name=f"dropout_{i}"
                    )
                )

        # Output layer
        model.add(
            tf.keras.layers.Dense(
                NUM_CLASSES,
                activation="softmax",
                kernel_initializer="glorot_uniform",
                bias_initializer="zeros",
                name="output",
            )
        )

        return model

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 20,
        batch_size: int = 32,
        verbose: int = 0,
    ) -> Dict[str, List[float]]:
        """
        Train the model on a task.

        Args:
            X_train: Training data (N, 784)
            y_train: Training labels (N, 10) one-hot encoded
            X_val: Validation data (optional)
            y_val: Validation labels (optional)
            epochs: Number of training epochs
            batch_size: Batch size
            verbose: TensorFlow verbose level

        Returns:
            Dictionary with training history

        Stores history in self.history for later analysis.
        """

        history = self.model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val) if X_val is not None else None,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
            shuffle=True,
        )

        # Append to running history
        self.history["losses"].extend(history.history["loss"])
        self.history["accuracies"].extend(history.history["accuracy"])

        if X_val is not None:
            self.history["val_losses"].extend(history.history["val_loss"])
            self.history["val_accuracies"].extend(history.history["val_accuracy"])

        return history.history

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Tuple[float, float]:
        """
        Evaluate model on test data.

        Args:
            X_test: Test data (N, 784)
            y_test: Test labels (N, 10) one-hot encoded

        Returns:
            (loss, accuracy) tuple
        """
        loss, accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        return float(loss), float(accuracy)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Get model predictions.

        Args:
            X: Input data (N, 784)

        Returns:
            Predictions of shape (N, 10)
        """
        return self.model.predict(X, verbose=0)

    def get_weights(self) -> List[np.ndarray]:
        """Get all model weights."""
        return self.model.get_weights()

    def set_weights(self, weights: List[np.ndarray]) -> None:
        """Set model weights."""
        self.model.set_weights(weights)

    def save_weights(self, filepath: str) -> None:
        """Save model weights to file."""
        self.model.save_weights(filepath)

    def load_weights(self, filepath: str) -> None:
        """Load model weights from file."""
        self.model.load_weights(filepath)
