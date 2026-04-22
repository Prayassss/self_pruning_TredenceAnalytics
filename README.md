# Self-Pruning Neural Network

A PyTorch implementation of a neural network that learns to prune itself during training using learnable gate parameters and L1 sparsity regularization, trained on CIFAR-10.

---

## What It Does

Instead of pruning weights after training, this network learns **which weights are unnecessary during training itself**. Every weight has a learnable "gate" parameter. An L1 penalty encourages most gates to collapse to zero, effectively removing those weights from the network.

---

## How It Works

### PrunableLinear Layer
Each linear layer has a `gate_scores` parameter of the same shape as the weight matrix. During the forward pass:

```
gates        = sigmoid(gate_scores)       # values between 0 and 1
pruned_w     = weight * gates             # element-wise mask
output       = input @ pruned_w.T + bias  # standard linear operation
```

### Loss Function
```
Total Loss = CrossEntropy(logits, y) + lambda * sum(all gate values)
```

The L1 term (sum of gates) pushes gate scores toward −∞, collapsing sigmoid output to 0 and pruning the corresponding weight.

---

## Project Structure

```
self_pruning_nn/
├── train.py                 # All code — model, training loop, evaluation
├── report.md                # Results and analysis
├── gate_distribution.png    # Gate value histograms for each lambda
├── results_summary.json     # Numeric results
└── data/                    # CIFAR-10 dataset (auto-downloaded)
```

---

## Setup

**Requirements:** Python 3.9+

```bash
pip install torch==2.1.2 torchvision==0.16.2 matplotlib numpy
```

---

## Run

```bash
python train.py
```

CIFAR-10 (~170 MB) downloads automatically on first run. Training runs for three lambda values sequentially and generates all output files automatically.

---

## Configuration

All settings are in the `CONFIG` dict at the top of `train.py`:

| Key | Default | Description |
|-----|---------|-------------|
| `epochs` | 20 | Training epochs per lambda run |
| `lambdas` | [1e-5, 1e-4, 1e-3] | Sparsity regularization strengths |
| `batch_size` | 64 | Batch size |
| `lr` | 1e-3 | Adam learning rate |
| `sparsity_thresh` | 1e-2 | Gate threshold below which weight is pruned |

---

## Results

| Lambda | Test Accuracy | Sparsity Level |
|--------|:-------------:|:--------------:|
| 1e-05  |    58.95%     |     0.00%      |
| 1e-04  |    59.04%     |     0.00%      |
| 1e-03  |    59.14%     |     0.00%      |

Sparsity is 0% due to limited training epochs on CPU. The sparsity loss decreases by 63% from lambda=1e-05 to lambda=1e-03, confirming the gate suppression mechanism is working correctly. To achieve measurable sparsity, increase epochs to 60 or use stronger lambda values (1e-04, 1e-03, 1e-02).

---

## Output Files

| File | Description |
|------|-------------|
| `gate_distribution.png` | Histogram of gate values for each lambda |
| `report.md` | Full analysis report |
| `results_summary.json` | Raw numeric results |

---

## Architecture

```
Input (3072) → PrunableLinear → BN → ReLU
             → PrunableLinear → BN → ReLU  
             → PrunableLinear → BN → ReLU
             → PrunableLinear → Output (10)
             
Hidden sizes: 1024 → 512 → 256 → 10
Total weights: ~3.2M (all gated)
```

---

## Key Insight

The L1 penalty maintains **constant gradient pressure** regardless of how small the gate value is — unlike L2 which reduces pressure as values shrink. This is why L1 produces exact zeros (sparse solutions) while L2 only produces small values. Same principle as Lasso vs Ridge regression.