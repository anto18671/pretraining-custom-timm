# pretraining-custom-timm

A flexible and extensible **PyTorch pretraining script** built atop the `timm` library. Designed for pretraining custom vision models with modern techniques like mixed precision, advanced augmentations, learning rate scheduling, EMA, and checkpointing. Updated **August 31, 2025**.

---

## Features

- **Model Configuration**: Leverages `timm` for easily switching architectures.
- **Data Loading & Augmentation**: Includes standard and advanced augment techniques (e.g., RandAugment, RandomErasing).
- **Mixed Precision Training**: Built-in support for AMP using PyTorch's `torch.amp`.
- **Learning Rate Scheduling**: Configurable warmup + cosine decay.
- **EMA (Exponential Moving Average)**: Maintains smoothed model weights for robust validation performance.
- **Checkpoint Strategy**: Saves robust checkpoints: `last.pth`, `best.pth`, `last_ema.pth`, and `best_ema.pth`.
- **Training Logging**: Progress tracking with `tqdm`; summary of model via `torchsummary`.

---

## Installation

```bash
git clone https://github.com/anto18671/pretraining-custom-timm.git
cd pretraining-custom-timm
pip install -r requirements.txt
```

**Requirements** (aligning with this repo’s `requirements.txt`):

- `torch`
- `torchvision`
- `timm`
- `tqdm`
- `torchsummary`
- `datasets` _(if Hugging Face datasets are used)_
  _(Adjust based on what's listed in the actual `requirements.txt`.)_

---

## Usage

Run the training script:

```bash
python train.py
```

### Default Hyperparameters

```text
Image size: 256
Batch size: 128
Epochs: 300
Warmup epochs: 20
Base LR: scaled as 5e-4 × (batch_size / 512)
Weight decay: 0.05
Mixup alpha: 0.8
CutMix alpha: 1.0
EMA decay: 0.9999
```

_(Update these defaults if your script uses different values.)_

---

## Checkpoints & Logging

- **Checkpoints**:

  - `last.pth`: Last epoch weights.
  - `best.pth`: Best validation performance (raw model).
  - `last_ema.pth`: EMA model at last epoch.
  - `best_ema.pth`: EMA model at best validation accuracy.

- **Training Logs**: Displayed via `tqdm` (loss, accuracy, learning rate).

---

## Example Training Console Output

```text
Epoch 001/300 | train_loss=5.4321 train_acc=0.0123 | \
val_loss=5.4210 val_acc=0.0145 | \
ema_val_loss=5.4100 ema_val_acc=0.0152 | \
best_acc=0.0145 best_ema_acc=0.0152 | lr=0.000012
```

---

## Additional Usage Examples

### Resuming from a Checkpoint

To resume training from the best checkpoint:

```python
checkpoint = torch.load("best.pth")
model.load_state_dict(checkpoint['state_dict'])
start_epoch = checkpoint['epoch'] + 1
```

### Evaluating EMA Weights

```python
ema_checkpoint = torch.load("best_ema.pth")
ema_model.load_state_dict(ema_checkpoint['state_dict'])
# Proceed with validation using ema_model...
```

---

## Future Extensions

- Distributed training with PyTorch (`torch.distributed`).
- Logging with TensorBoard or Weights & Biases.
- Fine-tuning scripts with custom head layers.
- Additional augmentations and data sampling strategies.

---

## License

This project is licensed under the **MIT License** — see the `LICENSE` file for details.
