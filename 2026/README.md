# Backdoor Attack + Adversarial Training Pipeline

PyTorch >= 2.6 pipeline for studying whether adversarially robust classifiers
resist backdoor (data poisoning) attacks. Supports MNIST, Fashion-MNIST,
CIFAR-10, and CIFAR-100 out of the box, plus pretrained robust models from
[RobustBench](https://robustbench.github.io/).

## Pipeline overview

```
1. Poison dataset         2. Train / load model           3. Evaluate
   (backdoor.py)             (model.py / RobustBench)        (main.py)

 clean images ──> stamp ──> poisoned train set ──> train SmallCNN/WRN   ──> accuracy
                  trigger    (alpha % flipped        (standard or           robust loss
                  pattern     to target label)        adversarial)          robust error
                                                  OR                        backdoor success
                                                  load pretrained  ─────>
                                                  robust model
                                                  (eval only or finetune)
```

**Core question:** If an adversarially trained model is robust to
L-inf perturbations, does that robustness also protect it against
pattern-based backdoor triggers injected into the training data?

## File structure

| File | Purpose |
|------|---------|
| `main.py` | CLI entry point, dataset loading, evaluation loops, TensorBoard logging |
| `attacks.py` | FGSM and PGD with chunked parallel restarts and perturbation-set assertions |
| `backdoor.py` | Trigger stamping (pixel / X-pattern / L-pattern) and dataset poisoning |
| `model.py` | `SmallCNN` architecture, training loop with adversarial training and mid-training eval |
| `wide_resnet.py` | WideResNet architecture with SiLU activations (Bartoldson et al., 2024) |
| `requirements.txt` | Dependencies (PyTorch, torchvision, robustbench, tensorboard, tqdm) |

## Quick start

```bash
pip install -r requirements.txt

# Train a standard SmallCNN on clean MNIST
python main.py --dataset mnist

# Poison 15% of CIFAR-10, adversarial training with WRN-28-10, target label 3
python main.py --dataset cifar10 --alpha 0.15 --adv-train --target 3 \
    --arch wrn-28-10 --pgd-eps 0.03137 --pgd-alpha 0.003 --pgd-iter 10 \
    --epochs 100 --batch-size 128 --lr 0.1

# Eval-only with a pretrained robust model (no training)
python main.py --dataset cifar10 --alpha 0.15 --robustbench Wong2020Fast \
    --pgd-eps 0.03137 --pgd-alpha 0.003

# Finetune a pretrained robust model on backdoored data
python main.py --dataset cifar10 --alpha 0.15 --target 3 \
    --robustbench Wong2020Fast --finetune --adv-train \
    --pgd-eps 0.03137 --pgd-alpha 0.003 --pgd-iter 10 \
    --epochs 10 --batch-size 128 --lr 1e-4

# Full sweep across poison rates
python main.py --dataset cifar10 --sweep --target 3 --robustbench Wong2020Fast

# Reproduce the original TF/MNIST results
for t in 0 1 2 3 4 5 6 7 8 9; do
    python main.py --dataset mnist --sweep --target $t \
        --style pattern --color 0.3 \
        --pgd-eps 0.3 --pgd-alpha 0.01 --pgd-iter 40 --pgd-restarts 10 \
        --epochs 2 --batch-size 32 --lr 1e-3
done

# View results
tensorboard --logdir runs
```

## Configurable hyperparameters

All set via CLI flags.  Run `python main.py --help` for the full list.

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset` | `mnist` | `mnist`, `fashion_mnist`, `cifar10`, `cifar100` |
| `--alpha` | `0.0` | Fraction of training set to poison |
| `--target` | `0` | Backdoor target label |
| `--style` | `pattern` | Trigger shape: `pixel`, `pattern` (X), `ell` (L) |
| `--color` | `0.3` | Trigger pixel intensity (0-1) |
| `--backdoor-eps` | - | Clamp trigger perturbation to Linf <= eps (e.g. 0.03137 for 8/255) |
| `--source-label` | `-1` | Which class to poison (-1 = all non-target) |
| `--arch` | `small` | `small` for SmallCNN, or `wrn-D-W` (e.g. `wrn-28-10`, `wrn-70-16`) |
| `--adv-train` | off | Enable PGD adversarial training |
| `--robustbench MODEL` | - | Load pretrained robust model from RobustBench |
| `--finetune` | off | Continue training a `--robustbench` model on the poisoned set |
| `--pgd-eps` | `0.3` | L-inf perturbation budget for PGD |
| `--pgd-alpha` | `0.01` | PGD step size |
| `--pgd-iter` | `40` | PGD steps per restart |
| `--pgd-restarts` | `10` | Number of random restarts |
| `--epochs` | `2` | Training epochs |
| `--batch-size` | `32` | Batch size (also controls batch-aligned poison count) |
| `--lr` | `1e-3` | Learning rate |
| `--eval-every N` | `0` | Run full robust eval every N epochs during training |
| `--eval-device` | - | Run mid-training eval on a second GPU asynchronously (e.g. `cuda:1`) |
| `--sweep` | off | Run full alpha x training-mode sweep |

## Execution modes

### 1. Train from scratch (default)

Trains a `SmallCNN` or `WideResNet` on the (possibly poisoned) dataset, then
evaluates accuracy, robust loss, robust error, and backdoor success rate on
both train and test sets.  `--adv-train` augments each batch with PGD
adversarial examples during training.

### 2. Pretrained eval (`--robustbench`)

Downloads a robust model from RobustBench, poisons the training set, and
reports accuracy + robust loss + robust error on the backdoored training set
and clean test set.  No training occurs.

Popular CIFAR-10 Linf models: `Wong2020Fast`, `Carmon2019Unlabeled`,
`Rice2020Overfitting`, `Engstrom2019Robustness`,
`Rebuffi2021Fixing_70_16_cutmix_extra`,
`Bartoldson2024Adversarial_WRN-94-16`.

### 3. Finetune pretrained (`--robustbench --finetune`)

Downloads a RobustBench model, then continues training (adversarial or
vanilla) on the poisoned dataset for `--epochs` steps.  Tests whether a
model that's already robust can be subverted by further training on poisoned
data.

## Evaluation metrics

All metrics are computed at the end of training and optionally during
training (`--eval-every`).

| Metric | Definition |
|--------|------------|
| **accuracy** | Fraction correctly classified on clean images |
| **binary_loss** | 1 - accuracy |
| **robust_loss** | Average worst-case cross-entropy: `(1/m) sum max_{x' in B(x,eps)} L(f(x'), y)` |
| **robust_error** | 1 - robust accuracy (fraction misclassified after PGD) |
| **backdoor_success** | Fraction of non-target test images classified as target after stamping |

On the **training set**, robust_loss and robust_error are measured against the
**poisoned labels** (including backdoor target labels).  On the **test set**,
they are measured against clean ground-truth labels.

## Poisoning logic

The number of poisoned examples is batch-aligned to match the original TF
implementation:

```
num_batches_to_add = int((alpha / (1 - alpha)) * (N / batch_size))
n_poison = num_batches_to_add * batch_size
```

Poisoned examples are **appended** to the clean training set (not replaced).
For alpha=0.15, N=60000, batch_size=32: 330 batches x 32 = 10,560 poisoned
examples appended to 60,000 clean ones.

When `--backdoor-eps` is set, trigger pixels are perturbed towards `--color`
by at most eps (Linf-bounded), rather than being overwritten absolutely.

## TensorBoard logging

Each run creates a unique timestamped directory under `runs/` containing:

- `hparams.json` -- all hyperparameters for reproducibility
- **Training scalars** (per batch): `train/loss`, `train/clean_loss`,
  `train/robust_loss`
- **Mid-training eval** (every `--eval-every` epochs):
  `eval/train_misclf_rate`, `eval/train_robust_loss`, `eval/train_robust_error`,
  `eval/val_misclf_rate`, `eval/val_robust_loss`, `eval/val_robust_error`
- **Final eval scalars**: `eval/train_accuracy`, `eval/train_robust_loss`,
  `eval/train_robust_error`, `eval/test_accuracy`, `eval/test_robust_loss`,
  `eval/test_robust_error`, `eval/backdoor_success`
- **Image grids**: `images/clean` (with labels) and `images/backdoored`
  (with original and target labels)

Disable with `--no-tensorboard`.  Change directory with `--logdir`.

## Model architectures

### SmallCNN (`--arch small`, default)

Lightweight CNN matching the original TF implementation.  Adapts to any
input shape via a dummy forward pass for flattened feature size.

### WideResNet (`--arch wrn-D-W`)

Pre-activation WideResNet with SiLU activations following Bartoldson et al.
(2024).  Supports all configurations from Table 2 of the paper:

| Depth | Width | Params | Flag |
|-------|-------|--------|------|
|  28   |   4   |   6M   | `wrn-28-4` |
|  28   |  10   |  36M   | `wrn-28-10` |
|  28   |  12   |  53M   | `wrn-28-12` |
|  70   |  16   | 267M   | `wrn-70-16` |
|  82   |  16   | 316M   | `wrn-82-16` |

Any `wrn-D-W` works as long as `(D - 4) % 6 == 0`.

## Performance engineering

The bottleneck is PGD: `restarts x num_iter` forward+backward passes per
batch.  Four techniques address this:

### 1. Chunked parallel restarts (`attacks.py`)

Instead of running 10 restarts sequentially, the batch is tiled
`(B, C, H, W)` -> `(B*P, C, H, W)` and P restarts run in a single forward
pass.  Restarts are processed in chunks of `parallel_restarts` (default 3)
to avoid OOM on large models:

- `restarts=10, parallel_restarts=3` -> 4 chunks (3+3+3+1)
- Each chunk does one forward pass on `B*3` images instead of 3 sequential
  passes on `B` images
- Best adversarial example per sample is tracked across chunks

Tune `parallel_restarts` based on your GPU memory.  Lower (1-2) for large
WideResNets on 40GB GPUs, higher (5-10) for small models or 80GB GPUs.

Additionally, **model parameters are frozen** during PGD (`requires_grad_(False)`)
since only input gradients are needed.  This reduces backward-pass memory
by not storing weight gradients.

Assertions verify that all adversarial examples satisfy the Linf perturbation
bound and [0, 1] pixel range.

### 2. `torch.compile` (`model.py`)

All models (SmallCNN, WideResNet, RobustBench, checkpoints) are wrapped with
`torch.compile`, which generates fused Triton kernels automatically for
conv/relu/sign/clamp chains.  This eliminates kernel launch overhead and
fuses elementwise operations without hand-written Triton.  Falls back
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

### 4. Async eval on second GPU (`--eval-device`)

When `--eval-device cuda:1` is set, mid-training evaluation runs
asynchronously in a background thread on a second GPU while training
continues on the primary device.  The model weights are snapshotted to CPU
and copied to the eval device, so training is not blocked.  If the next
eval checkpoint arrives before the previous eval finishes, it waits.
