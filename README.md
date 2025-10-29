# Catastrophic Forgetting in Neural Networks: CS 599 Deep Learning Analysis

## Overview

This repository contains a comprehensive analysis of **catastrophic forgetting** in multi-layer perceptrons trained on permuted MNIST. The code implements the continual learning evaluation framework from [Gradient Episodic Memory (GEM, NIPS 2017)](https://arxiv.org/abs/1706.08840) and extends it to analyze how different architectural choices, loss functions, and optimization strategies affect the model's ability to learn new tasks without degrading performance on previously learned tasks.

### What is Catastrophic Forgetting?

Catastrophic forgetting (also called "catastrophic interference") is a fundamental problem in continual learning:

**Scenario:**

1. Train model on Task A → achieves 90% accuracy
2. Train same model on Task B → learns Task B well
3. Test on Task A → accuracy drops to 20% 😱

The model has "forgotten" Task A while learning Task B. This happens because learning Task B overwrites the weights that were optimized for Task A.

**This project quantifies and analyzes this phenomenon across multiple experimental conditions.**

---

## Architecture & Design Decisions

### Why This Architecture?

The codebase is organized into modular components with clear separation of concerns:

```
catastrophic_forgetting/
├── config.py                    # Single source of truth for all parameters
├── utils/
│   ├── data_pipeline.py        # Data loading & preprocessing (cacheable)
│   ├── metrics.py              # Metric computation (GEM framework)
│   └── plotting.py             # Publication-quality visualizations
├── models/
│   └── mlp.py                  # MLP architecture & training
├── experiments/
│   └── runner.py               # Experiment orchestration & caching
├── solution.py                 # Main entry point
└── results/                    # Cached data & generated figures
```

**Key Design Principles:**

1. **Single Source of Truth (config.py)**
   - All hyperparameters, paths, and experiment definitions in one file
   - Easy to modify experiments without touching code
   - Reproducible across runs

2. **Intelligent Caching**
   - Permuted MNIST generated once, reused across 42 experiments
   - Experiment results cached by config hash
   - 2.19 GB one-time cost saves 3+ hours on subsequent runs

3. **Functional Separation**
   - Data: Permuted MNIST generation, preprocessing, caching
   - Models: Architecture definition, training, evaluation
   - Experiments: Orchestration, result aggregation, caching
   - Visualization: Publication-ready plotting with consistent styling

4. **Reproducibility**
   - Deterministic seed generation from "Karl"
   - All random number generators seeded
   - Identical parameters → identical results

---

## Quick Start

### 1. Installation

```bash
# Clone repository
git clone <url>
cd catastrophic_forgetting

# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# [Mac with Apple Silicon] Verify GPU support
python -c "import tensorflow as tf; print(f'GPUs: {tf.config.list_physical_devices(\"GPU\")}')"
```

### 2. Run Complete Pipeline

```bash
# Run all 42 experiments (first time: 3-5 hours, subsequent: 30 seconds)
python solution.py

# Run with progress output
python solution.py --verbose-tf 1

# Force re-run all experiments (ignore cache)
python solution.py --force-regenerate

# Suppress debug logging
python solution.py --quiet
```

### 3. Output

After execution, find results in `results/`:

```
results/
├── cache/
│   ├── permuted_mnist/              # Cached MNIST data
│   └── experiments/                 # Cached experiment results
├── figures/
│   ├── loss_comparison.pdf
│   ├── dropout_effect.pdf
│   ├── depth_analysis.pdf
│   ├── optimizer_comparison.pdf
│   ├── validation_curves.pdf
│   ├── acc_vs_bwt_scatter.pdf
│   ├── cumulative_forgetting.pdf
│   ├── task_matrices_heatmap.pdf
│   ├── metrics_distribution_boxplots.pdf
│   └── metrics_summary.json
└── results_table.tex                # LaTeX-ready results table
```

---

## Experiments Explained

### The Task Matrix R

The core data structure for evaluating continual learning. R is a 10×10 matrix where:

- **R[i, j]** = Accuracy on Task j after training through Task i

Example interpretation:

```
         Task 0   Task 1   Task 2   ...
       ┌──────────────────────────────┐
Task 0 │ 0.92     0.10     0.05       │  (After learning Task 0)
Task 1 │ 0.85     0.90     0.10       │  (After learning Tasks 0-1)
Task 2 │ 0.42     0.88     0.91       │  (After learning Tasks 0-2)
...    │ ...      ...      ...        │
Task 9 │ 0.30     0.65     0.78       │  (After learning all tasks)
       └──────────────────────────────┘

Observations:
- Diagonal: High values (good on just-learned tasks)
- Lower-left: Low values (forgetting!) – 92% → 30% on Task 0
- This lower-left triangle visualizes catastrophic forgetting
```

### The Metrics (from GEM Paper)

**ACC (Average Accuracy)**

- Formula: `mean(R[9, :])`
- What: Overall competence after learning all tasks
- Range: [0, 1], higher is better
- Example: 0.65 means 65% average accuracy across all tasks

**BWT (Backward Transfer)** ⭐ Most Important

- Formula: `mean(R[9, j] - R[j, j])` for j in 0..8
- What: Average degradation on previous tasks
- Interpretation:
  - BWT = -0.35 → Lost 35% accuracy on old tasks (severe forgetting)
  - BWT = 0 → No forgetting
  - BWT = +0.05 → Gained 5% on old tasks (positive transfer)
- Range: [-1, 1], higher is better

**FWT (Forward Transfer)**

- Formula: `mean(R[i, i+1] - baseline)` for i in 0..8
- What: Does learning previous tasks help learn new tasks?
- Range: [-1, 1], higher is better

### Experiment Suites

#### Experiment 1: Loss Function Comparison (12 experiments)

**Question:** Does the loss function affect catastrophic forgetting?

**Configuration:**

- Varying: Loss ∈ {NLL, L1, L2, L1+L2}, Depth ∈ {2, 3, 4}
- Fixed: Dropout=0.0, Optimizer=Adam

**Expected Findings:**

- NLL (baseline) should perform reasonably
- L1/L2 regularization might help or hurt forgetting

**Visualization:** `loss_comparison.pdf`

#### Experiment 2: Dropout Effect (9 experiments)

**Question:** Does dropout regularization help prevent forgetting?

**Configuration:**

- Varying: Dropout ∈ {0.0, 0.2, 0.5}, Depth ∈ {2, 3, 4}
- Fixed: Loss=NLL, Optimizer=Adam

**Expected Findings:**

- Heavy dropout (0.5) might prevent overfitting but could hurt learning
- Optimal dropout rate balances learning and regularization

**Visualization:** `dropout_effect.pdf`

#### Experiment 3: Depth Analysis (3 experiments)

**Question:** Does network depth affect catastrophic forgetting?

**Configuration:**

- Varying: Depth ∈ {2, 3, 4}
- Fixed: Loss=NLL, Dropout=0.0, Optimizer=Adam

**Expected Findings:**

- Deeper networks have more capacity (higher ACC?)
- May be harder to train on task sequences (worse BWT?)
- Optimal depth for continual learning may differ from standard learning

**Visualization:** `depth_analysis.pdf`

#### Experiment 4: Optimizer Comparison (9 experiments)

**Question:** Does the optimizer affect continual learning performance?

**Configuration:**

- Varying: Optimizer ∈ {SGD, Adam, RMSprop}, Depth ∈ {2, 3, 4}
- Fixed: Loss=NLL, Dropout=0.0

**Expected Findings:**

- Adam typically converges faster (better initial learning)
- SGD with momentum may generalize better (less forgetting?)
- RMSprop is a middle ground

**Visualization:** `optimizer_comparison.pdf`

#### Experiment 5: Validation Curves (3 experiments)

**Question:** How does performance on Task 0 degrade as we learn Tasks 1-9?

**Configuration:**

- Track: Accuracy on Task 0 after each task training
- Compare: Depth ∈ {2, 3, 4}

**Expected Findings:**

- Steep decline in Task 0 accuracy = catastrophic forgetting
- Shallower decline = better continual learning
- Different depths may show different forgetting rates

**Visualization:** `validation_curves.pdf`

---

## Understanding the Code

### Key Classes & Functions

#### `ExperimentConfig` (config.py)

Defines a single experiment:

```python
config = ExperimentConfig(
    depth=2,                      # 2 hidden layers
    loss_type="nll",              # Categorical cross-entropy
    dropout_rate=0.0,             # No dropout
    optimizer="adam",             # Adam optimizer
    experiment_name="baseline"
)

# Get unique identifier for caching
hash_id = config.get_config_hash()  # "a1b2c3d4..."

# Get human-readable filename
name = config.get_readable_name()  # "depth2_nll_dropout0p0_adam"
```

#### `load_permuted_task_data()` (data_pipeline.py)

Load permuted MNIST for a specific task:

```python
(X_train, y_train), (X_test, y_test) = load_permuted_task_data(
    task_id=3,  # Different pixel permutation per task
    seed=SEED_VALUE
)
# X_train: (60000, 784) float32, values in [0, 1]
# y_train: (60000, 10) one-hot encoded
```

#### `MLPClassifier` (models/mlp.py)

Build and train a model:

```python
model = MLPClassifier(
    depth=3,
    dropout_rate=0.2,
    loss_type="nll",
    optimizer_type="adam"
)

# Train on task
model.fit(X_train, y_train, epochs=50, batch_size=32)

# Evaluate
loss, accuracy = model.evaluate(X_test, y_test)

# Get predictions
predictions = model.predict(X_test)  # Shape: (N, 10)
```

#### `TaskMatrix` (metrics.py)

Record task matrix R:

```python
matrix = TaskMatrix(num_tasks=10)

# After training each task
for task_id in range(10):
    accuracies = model.evaluate_on_all_tasks()  # Shape: (10,)
    matrix.record_accuracies(task_id, accuracies)  # Fills row i

# Compute metrics
metrics = compute_all_metrics(matrix.R)
# Returns: {'acc': 0.65, 'bwt': -0.35, 'fwt': 0.02, ...}
```

#### `ExperimentRunner` (experiments/runner.py)

Orchestrate experiments with caching:

```python
runner = ExperimentRunner(seed=SEED_VALUE, force_regenerate=False)

# Run single experiment (with caching)
result = runner.run_experiment(config, verbose=0)
print(f"ACC: {result.metrics['acc']:.4f}")

# Run multiple experiments
results = runner.run_experiment_suite([config1, config2, ...])
```

---

## How Caching Works

### Why Caching?

Running all 42 experiments from scratch takes **3-5 hours**. Caching enables:

- Subsequent full runs in **30 seconds** (all cache hits)
- Regeneration on visualization without re-running experiments
- Safe interruption and resumption

### Two-Level Cache

**Level 1: Permuted MNIST Data**

- Location: `results/cache/permuted_mnist/task_*.pkl`
- Size: 2.19 GB total (one-time cost)
- First load: ~60 seconds (MNIST download + processing)
- Subsequent: ~1 second (load from pickle)
- Shared across all 42 experiments

**Level 2: Experiment Results**

- Location: `results/cache/experiments/result_<hash>.json`
- Key: MD5 hash of ExperimentConfig
- First run: 60-90 minutes per experiment
- Subsequent: <1 second per experiment

### Cache Invalidation

Cache is automatically invalidated when:

1. Config parameters change (different hash)
2. `force_regenerate=True` flag passed
3. Cache files are manually deleted

---

## Performance Characteristics

### First Run (Clean Cache)

```
Data loading:             ~ 60 seconds
Training 42 experiments:  ~ 45 hours
Visualization:            ~  5 minutes
Total:                    ~  2 days 
```

### Subsequent Runs (Cache Hits)

```
Data loading:             ~ 1 second
All experiments:          < 1 second (JSON load)
Visualization:            ~ 2 minutes
Total:                    ~ 2 minutes
```

### System Requirements

- **Disk Space:** 2.5 GB (MNIST cache + results)
- **RAM:** 4 GB minimum (16 GB recommended)
- **GPU:** Optional but recommended (Metal on Mac, CUDA on Linux)
- **Time:** 2 days first run, 2 minutes cached

---

## Interpreting Results

### Bad BWT (Severe Forgetting)

```
BWT = -0.50  →  Lose 50% accuracy on old tasks

Example:
- Task 0: 90% immediately after learning
- Task 0: 40% after learning 9 more tasks
- Catastrophic forgetting confirmed
```

### Good BWT (Minimal Forgetting)

```
BWT = -0.05  →  Lose only 5% accuracy on old tasks

Example:
- Task 0: 90% immediately after learning
- Task 0: 85% after learning 9 more tasks
- Continual learning is working reasonably well
```

### Comparing Metrics Across Configurations

**Higher ACC is better:**

- 0.9: Excellent all-around performance
- 0.7: Good performance
- 0.5: Moderate performance
- <0.3: Poor

**Higher BWT is better (less negative = less forgetting):**

- BWT > 0: Positive transfer (rare and good!)
- BWT ≈ 0: No forgetting (ideal)
- BWT = -0.3: Moderate forgetting (expected)
- BWT < -0.5: Severe forgetting (problem!)

**Task-by-task Analysis:**
Look at task matrix R to see which tasks suffer most:

```
Task 0: 92% → 30% (62% degradation) – Early tasks hit hard
Task 5: 88% → 72% (16% degradation) – Middle tasks affected less
Task 9: 92% → 92% (0% degradation) – Latest task unchanged
```

---

## Extending the Code

### Adding New Experiments

1. **Define new config in `config.py`:**

```python
new_config = ExperimentConfig(
    depth=2,
    loss_type="l2",
    dropout_rate=0.3,  # New value!
    optimizer="adam",
    experiment_name="custom_experiment"
)
```

2. **Add to experiment suite:**

```python
def build_experiment_suite():
    # ... existing code ...
    custom_experiments = [new_config]
    
    return {
        # ... existing suites ...
        "custom": custom_experiments,
    }
```

3. **Run and visualize:**

```bash
python solution.py --force-regenerate  # Test new config
```

### Adding New Metrics

1. **Implement in `metrics.py`:**

```python
def compute_new_metric(R: np.ndarray) -> float:
    """My new metric..."""
    # Compute from task matrix R
    return result

def compute_all_metrics(R: np.ndarray) -> Dict[str, float]:
    # ... existing metrics ...
    return {
        'acc': compute_acc(R),
        'bwt': compute_bwt(R),
        'new_metric': compute_new_metric(R),  # Add here
    }
```

2. **Visualize in `plotting.py`:**

```python
def plot_new_metric(results):
    # Create figure comparing new_metric across configs
    pass

# Call in generate_all_visualizations():
figure_paths['new_metric'] = plot_new_metric(results)
```

---

## Technical Details: Metal GPU Support

### On Apple Silicon (M1/M2/M3)

TensorFlow 2.16+ includes native Metal GPU support:

```bash
# Install with Metal support
pip install tensorflow-metal

# Verify GPU is being used
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
# Expected output: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

If not detected:

```bash
# Reinstall without cache
pip install --no-cache-dir tensorflow-metal tensorflow

# Verify Metal plugin
python -c "from tensorflow import keras; print(keras.backend.device)"
```

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'tensorflow'"

**Solution:**

```bash
pip install -r requirements.txt
```

### Issue: Experiments running very slowly

**Solution:** Verify GPU is being used:

```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices())"
```

### Issue: Cache is corrupted

**Solution:** Clear cache and re-run:

```bash
rm -rf results/cache/
python solution.py
```

### Issue: Out of memory errors

**Solution:** Reduce batch size in `config.py`:

```python
BATCH_SIZE = 16  # From 32
```

---

## References

### Academic Papers

1. **Gradient Episodic Memory for Continual Learning**
   - Lopez-Paz, D., & Ranzato, M. (NeurIPS 2017)
   - Defines ACC, BWT, FWT metrics
   - Proposes GEM method to prevent forgetting
   - [Link](https://arxiv.org/abs/1706.08840)

2. **Lifelong Neural Predictive Coding: Learning Cumulatively Online without Forgetting**
   - Ororbia et al. (NeurIPS 2022)
   - Extends metrics with TBWT, CBWT
   - Proposes alternative approach to catastrophic forgetting
   - [Link](https://arxiv.org/abs/1905.10696)

### Related Resources

- Continual Learning: [Parisi et al. Survey](https://arxiv.org/abs/1802.07569)
- Catastrophic Forgetting: [French (1999) Classic Paper](https://www.sciencedirect.com/science/article/pii/S1364661399012949)
- Permuted MNIST Benchmark: Standard for continual learning evaluation

---

## Questions?

Refer to docstrings in the code for detailed explanations of specific functions and classes. Each module is heavily documented to support understanding.

```bash
# View docstrings in Python
python -c "from config import ExperimentConfig; help(ExperimentConfig)"
python -c "from utils.metrics import compute_all_metrics; help(compute_all_metrics)"
```
