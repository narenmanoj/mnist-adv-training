# Backdoor Attack + Adversarial Training Pipeline

PyTorch >= 2.6 pipeline for studying whether adversarially robust classifiers
resist backdoor (data poisoning) attacks. Supports MNIST, Fashion-MNIST,
CIFAR-10, and CIFAR-100 out of the box, plus pretrained robust models from
[RobustBench](https://robustbench.github.io/).

## Pipeline overview

```
1. Poison dataset         2. Train / load model         3. Evaluate
   (backdoor.py)             (model.py / RobustBench)      (main.py)

 clean images ──> stamp ──> poisoned train set ──> train SmallCNN        ──> accuracy
                  trigger    (alpha % flipped        (standard or           robust loss
                  pattern     to target label)        adversarial)          backdoor success
                                                  OR
                                                  load pretrained ──────> accuracy
                                                  robust model            robust loss
                                                  (eval only)             backdoor success
```

**Core question:** If an adversarially trained model is robust to
L-inf perturbations, does that robustness also protect it against
pattern-based backdoor triggers injected into the training data?

## File structure

| File | Purpose |
|------|---------|
| `main.py` | CLI entry point, dataset loading, evaluation loops, TensorBoard logging |
| `attacks.py` | FGSM and PGD with chunked parallel restarts |
| `backdoor.py` | Trigger stamping (pixel / X-pattern / L-pattern) and dataset poisoning |
| `model.py` | `SmallCNN` architecture, training loop with optional adversarial training |
| `requirements.txt` | Dependencies (PyTorch, torchvision, robustbench, tensorboard, tqdm) |

## Quick start

```bash
pip install -r requirements.txt

# Train a standard model on clean MNIST
python main.py --dataset mnist

# Poison 15% of CIFAR-10, adversarial training, target label 3
python main.py --dataset cifar10 --alpha 0.15 --adv-train --target 3

# Eval-only with a pretrained robust model (no training)
python main.py --dataset cifar10 --alpha 0.15 --robustbench Wong2020Fast

# Full sweep across poison rates
python main.py --dataset cifar10 --sweep --target 3 --robustbench Wong2020Fast

# View results
tensorboard --logdir runs
```

## Configurable hyperparameters

All set via CLI flags. Key ones:

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset` | `mnist` | `mnist`, `fashion_mnist`, `cifar10`, `cifar100` |
| `--alpha` | `0.0` | Fraction of training set to poison |
| `--target` | `0` | Backdoor target label |
| `--style` | `pattern` | Trigger shape: `pixel`, `pattern` (X), `ell` (L) |
| `--color` | `0.3` | Trigger pixel intensity (0-1) |
| `--source-label` | `-1` | Which class to poison (-1 = all non-target) |
| `--adv-train` | off | Enable PGD adversarial training |
| `--robustbench MODEL` | - | Load pretrained robust model (skips training) |
| `--pgd-eps` | `0.3` | L-inf perturbation budget |
| `--pgd-iter` | `40` | PGD steps per restart |
| `--pgd-restarts` | `10` | Number of random restarts |
| `--epochs` | `2` | Training epochs |
| `--sweep` | off | Run full alpha x training-mode sweep |

Run `python main.py --help` for the complete list.

## Two execution modes

### Train from scratch (default)

Trains a `SmallCNN` on the (possibly poisoned) dataset, then evaluates
accuracy, robust error rate, and backdoor success rate on both train and
test sets. `--adv-train` augments each batch with PGD adversarial examples
during training.

### Pretrained eval (`--robustbench`)

Downloads a robust model from RobustBench, poisons the training set, and
reports accuracy + robust error on the backdoored training set and clean
validation set. No training occurs. Useful for quickly checking whether
known-robust models resist various backdoor configurations.

Popular CIFAR-10 Linf models: `Wong2020Fast`, `Carmon2019Unlabeled`,
`Rice2020Overfitting`, `Engstrom2019Robustness`,
`Rebuffi2021Fixing_70_16_cutmix_extra`.

## TensorBoard logging

Each run creates a timestamped directory under `runs/` containing:

- `hparams.json` -- all hyperparameters for reproducibility
- **Training scalars** (per batch): `train/loss`, `train/clean_loss`,
  `train/robust_loss`
- **Eval scalars**: `eval/train_accuracy`, `eval/train_robust_error`,
  `eval/val_accuracy`, `eval/val_robust_error`, `eval/backdoor_success`
- **Image grids**: `images/clean` (with labels) and `images/backdoored`
  (with original and target labels)

Disable with `--no-tensorboard`. Change directory with `--logdir`.

## Performance engineering

The bottleneck is PGD: `restarts x num_iter` forward+backward passes per
batch. Three techniques bring this down from hours to minutes:

### 1. Chunked parallel restarts (`attacks.py`)

Instead of running 10 restarts sequentially, the batch is tiled
`(B, C, H, W)` -> `(B*P, C, H, W)` and P restarts run in a single forward
pass. Restarts are processed in chunks of `parallel_restarts` (default 3)
to avoid OOM on large models:

- `restarts=10, parallel_restarts=3` -> 4 chunks (3+3+3+1)
- Each chunk does one forward pass on `B*3` images instead of 3 sequential
  passes on `B` images
- Best adversarial example per sample is tracked across chunks

Tune `parallel_restarts` based on your GPU memory. Lower (1-2) for large
WideResNets on 40GB GPUs, higher (5-10) for small models or 80GB GPUs.

Additionally, **model parameters are frozen** during PGD (`requires_grad_(False)`)
since only input gradients are needed. This reduces backward-pass memory
by not storing weight gradients.

### 2. `torch.compile` (`model.py`)

All models (SmallCNN, RobustBench, checkpoints) are wrapped with
`torch.compile`, which generates fused Triton kernels automatically for
conv/relu/sign/clamp chains. This eliminates kernel launch overhead and
fuses elementwise operations without hand-written Triton. Falls back
gracefully on unsupported platforms.

### 3. Mixed precision / AMP (`model.py`, `main.py`)

`torch.amp.autocast` is applied to all forward passes during training,
PGD, and evaluation:

- **CUDA**: bfloat16 if supported, else float16 with GradScaler
- **MPS**: float16
- **CPU**: disabled (no benefit)

This roughly doubles throughput on modern NVIDIA GPUs by using Tensor Cores
for matmuls and convolutions while keeping perturbation arithmetic in
float32.
