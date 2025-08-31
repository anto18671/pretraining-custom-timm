# ---------------------------
# Imports
# ---------------------------
import math
import os
import sys
from typing import Tuple

import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import timm
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy, LabelSmoothingCrossEntropy
from tqdm import tqdm
import torchsummary
from datasets import load_dataset
from huggingface_hub import snapshot_download, login


# ---------------------------
# Compose torchvision transforms with RandAugment and RandomErasing
# ---------------------------
def build_transforms(image_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
    train_tfms = transforms.Compose([
        transforms.Resize(int(image_size * 1.15), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomResizedCrop(image_size, scale=(0.08, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3), value='random', inplace=False)
    ])

    valid_tfms = transforms.Compose([
        transforms.Resize(int(image_size * 1.15), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    return train_tfms, valid_tfms


# ---------------------------
# Thin PyTorch Dataset wrapper around HF imagefolder splits
# ---------------------------
class HFImageFolder(Dataset):
    # Initialize with a split and a transform
    def __init__(self, hf_split, transform):
        self.data = hf_split
        self.transform = transform

    # Length of split
    def __len__(self):
        return len(self.data)

    # Get one (image, label) pair
    def __getitem__(self, idx):
        sample = self.data[idx]
        image = sample["image"].convert("RGB")
        label = sample["label"]
        image = self.transform(image)
        return image, label


# ---------------------------
# Detect split names inside a downloaded imagefolder repo
# ---------------------------
def detect_splits(local_dir: str) -> Tuple[str, str]:
    # List candidate names
    train_candidates = ["train", "training"]
    val_candidates = ["val", "valid", "validation"]

    # Initialize discovered names
    train_name = None
    val_name = None

    # Search for train-like split
    for name in train_candidates:
        if os.path.isdir(os.path.join(local_dir, name)):
            train_name = name
            break

    # Search for val-like split
    for name in val_candidates:
        if os.path.isdir(os.path.join(local_dir, name)):
            val_name = name
            break

    # Validate discovery
    if train_name is None or val_name is None:
        raise FileNotFoundError(
            f"Could not find expected splits under '{local_dir}'. "
            f"Expected one of {train_candidates} and one of {val_candidates} as subfolders."
        )

    return train_name, val_name


# ---------------------------
# Download dataset repo from Hugging Face Hub (no streaming)
# ---------------------------
def download_dataset_repo(repo_id: str, revision: str, token_env: str, local_cache_dir: str) -> str:
    # Login if token is present in environment
    if token_env and len(token_env.strip()) > 0:
        login(token=token_env.strip())

    # Snapshot the dataset repository locally
    local_dir = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision if revision else None,
        local_dir=local_cache_dir if local_cache_dir else None,
        local_dir_use_symlinks=False,
        allow_patterns=None,
        ignore_patterns=None
    )

    return local_dir


# ---------------------------
# Create DataLoaders from downloaded imagefolder dataset
# ---------------------------
def create_dataloaders_from_parquet(image_size: int, batch_size: int, data_dir: str):
    # Build transforms
    train_tfms, valid_tfms = build_transforms(image_size)

    # Build file patterns
    train_glob = os.path.join(data_dir, "train-*.parquet")
    val_glob   = os.path.join(data_dir, "validation-*.parquet")

    # Load local parquet datasets
    ds = load_dataset(
        "parquet",
        data_files={
            "train": train_glob,
            "validation": val_glob
        }
    )


    # Wrap with our thin torch Dataset
    train_ds = HFImageFolder(ds["train"], transform=train_tfms)
    val_ds   = HFImageFolder(ds["validation"], transform=valid_tfms)

    # Derive class mapping when available
    try:
        label_feature = ds["train"].features["label"]
        class_to_idx = {name: i for i, name in enumerate(label_feature.names)}
    except Exception:
        # Fallback if features lack names: infer max label id
        try:
            max_label = max(int(x["label"]) for x in ds["train"])
            class_to_idx = {str(i): i for i in range(max_label + 1)}
        except Exception:
            class_to_idx = {}

    # Worker heuristics for Windows
    num_workers_train = max(0, os.cpu_count() - 2) if os.name == "nt" else 8
    num_workers_val = max(1, num_workers_train // 2)

    # Build DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers_train,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(num_workers_train > 0)
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers_val,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(num_workers_val > 0)
    )

    return train_loader, val_loader, class_to_idx


# ---------------------------
# Build model, loss, optimizer, device, mixup/cutmix
# ---------------------------
def build_training_objects(num_classes: int, base_lr: float, weight_decay: float, mixup_alpha: float, cutmix_alpha: float):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = timm.create_model(
        "efficientvit_b3",
        pretrained=False,
        num_classes=num_classes,
        drop_rate=0.25,
        widths=(16, 32, 64, 128, 256),
        depths=(1, 3, 5, 5, 6),
        head_dim=16,
        head_widths=(1024, 1024)
    )

    model = model.to(device).to(memory_format=torch.channels_last)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=base_lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )

    train_criterion = SoftTargetCrossEntropy()
    eval_criterion = LabelSmoothingCrossEntropy(smoothing=0.1)

    mixup_fn = Mixup(
        mixup_alpha=mixup_alpha,
        cutmix_alpha=cutmix_alpha,
        prob=1.0,
        switch_prob=0.0,
        mode='batch',
        label_smoothing=0.0,
        num_classes=num_classes
    )

    return model, optimizer, train_criterion, eval_criterion, device, mixup_fn


# ---------------------------
# Build EMA model using PyTorch AveragedModel with exponential moving average
# ---------------------------
def build_ema_model(model: torch.nn.Module, ema_decay: float):
    # Define EMA averaging function
    def ema_avg_fn(averaged_param, current_param, num_averaged):
        return ema_decay * averaged_param + (1.0 - ema_decay) * current_param

    # Wrap model with AveragedModel
    ema_model = AveragedModel(model, avg_fn=ema_avg_fn, use_buffers=True)

    return ema_model


# ---------------------------
# Warmup-cosine scheduler builder
# ---------------------------
def build_warmup_cosine_scheduler(optimizer: optim.Optimizer, warmup_epochs: int, total_epochs: int, steps_per_epoch: int):
    # Define lr lambda across steps
    def lr_lambda(step: int):
        total_steps = total_epochs * steps_per_epoch
        warmup_steps = warmup_epochs * steps_per_epoch
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    # Create scheduler
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    return scheduler


# ---------------------------
# Train for one epoch with AugReg, AMP, and EMA updates
# ---------------------------
def train_one_epoch(model, ema_model, loader, train_criterion, optimizer, scheduler, device, scaler, mixup_fn, epoch, epochs, steps_done):
    model.train()

    running_loss = 0.0
    running_correct = 0
    running_count = 0

    loop = tqdm(loader, desc=f"Train Epoch {epoch}/{epochs}", leave=False)

    for images, targets in loop:
        images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
        targets = targets.to(device, non_blocking=True)

        images, targets = mixup_fn(images, targets)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
            outputs = model(images)
            loss = train_criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        ema_model.update_parameters(model)

        scheduler.step()
        steps_done += 1

        preds = outputs.float().softmax(dim=1).argmax(dim=1)
        hard_targets = targets.float().argmax(dim=1)

        running_loss += loss.item() * images.size(0)
        running_correct += (preds == hard_targets).sum().item()
        running_count += images.size(0)

        avg_loss = running_loss / max(1, running_count)
        avg_acc = running_correct / max(1, running_count)
        loop.set_postfix(loss=avg_loss, acc=avg_acc)

    epoch_loss = running_loss / max(1, running_count)
    epoch_acc = running_correct / max(1, running_count)

    return epoch_loss, epoch_acc, steps_done


# ---------------------------
# Evaluate on validation set
# ---------------------------
def evaluate(model, loader, eval_criterion, device, epoch, epochs):
    model.eval()

    running_loss = 0.0
    running_correct = 0
    running_count = 0

    loop = tqdm(loader, desc=f"Valid Epoch {epoch}/{epochs}", leave=False)

    with torch.no_grad():
        for images, targets in loop:
            images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
            targets = targets.to(device, non_blocking=True)

            outputs = model(images)
            loss = eval_criterion(outputs, targets)

            preds = outputs.argmax(dim=1)

            running_loss += loss.item() * images.size(0)
            running_correct += (preds == targets).sum().item()
            running_count += images.size(0)

            avg_loss = running_loss / max(1, running_count)
            avg_acc = running_correct / max(1, running_count)
            loop.set_postfix(loss=avg_loss, acc=avg_acc)

    epoch_loss = running_loss / max(1, running_count)
    epoch_acc = running_correct / max(1, running_count)

    return epoch_loss, epoch_acc


# ---------------------------
# Training orchestration with EMA and Hub auto-download (no streaming)
# ---------------------------
def main():
    cudnn.benchmark = True

    # Define dataset repo info
    repo_id = os.environ.get("HF_IMAGENET_REPO", "benjamin-paine/imagenet-1k-256x256")
    revision = os.environ.get("HF_IMAGENET_REVISION", None)

    # Define auth
    token_env = os.environ.get("HUGGINGFACE_TOKEN", "")

    # Define local cache dir
    local_cache_dir = os.environ.get("HF_DATA_CACHE_DIR", "./imagenet1k")

    # Define training hyperparameters
    image_size = 256
    batch_size = 128
    epochs = 300
    warmup_epochs = 20
    base_lr = 5e-4 * (batch_size / 512.0)
    weight_decay = 0.05
    mixup_alpha = 0.8
    cutmix_alpha = 1.0
    ema_decay = 0.9999

    # Path containing your parquet shards
    parquet_dir = os.path.join(local_cache_dir, "data")

    # Build loaders from local parquet
    try:
        train_loader, val_loader, class_to_idx = create_dataloaders_from_parquet(
            image_size=image_size,
            batch_size=batch_size,
            data_dir=parquet_dir
        )
    except Exception as e:
        print(f"[FATAL] Failed to build dataloaders from parquet in '{parquet_dir}'. Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Build training objects
    num_classes = len(class_to_idx)
    model, optimizer, train_criterion, eval_criterion, device, mixup_fn = build_training_objects(
        num_classes=num_classes,
        base_lr=base_lr,
        weight_decay=weight_decay,
        mixup_alpha=mixup_alpha,
        cutmix_alpha=cutmix_alpha
    )

    # Build EMA model
    ema_model = build_ema_model(model, ema_decay=ema_decay)
    ema_model = ema_model.to(device).to(memory_format=torch.channels_last)

    # Build AMP scaler
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # Print model summary
    torchsummary.summary(model, (3, image_size, image_size))

    # Build scheduler
    steps_per_epoch = len(train_loader)
    scheduler = build_warmup_cosine_scheduler(
        optimizer,
        warmup_epochs=warmup_epochs,
        total_epochs=epochs,
        steps_per_epoch=steps_per_epoch
    )

    # Initialize trackers
    best_acc = 0.0
    best_ema_acc = 0.0
    global_steps = 0

    # Train epochs
    for epoch in range(1, epochs + 1):
        train_loss, train_acc, global_steps = train_one_epoch(
            model=model,
            ema_model=ema_model,
            loader=train_loader,
            train_criterion=train_criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            scaler=scaler,
            mixup_fn=mixup_fn,
            epoch=epoch,
            epochs=epochs,
            steps_done=global_steps
        )

        val_loss, val_acc = evaluate(
            model=model,
            loader=val_loader,
            eval_criterion=eval_criterion,
            device=device,
            epoch=epoch,
            epochs=epochs
        )

        ema_val_loss, ema_val_acc = evaluate(
            model=ema_model,
            loader=val_loader,
            eval_criterion=eval_criterion,
            device=device,
            epoch=epoch,
            epochs=epochs
        )

        torch.save({'epoch': epoch, 'state_dict': model.state_dict()}, "last.pth")
        torch.save({'epoch': epoch, 'state_dict': ema_model.state_dict()}, "last_ema.pth")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'acc1': best_acc}, "best.pth")

        if ema_val_acc > best_ema_acc:
            best_ema_acc = ema_val_acc
            torch.save({'epoch': epoch, 'state_dict': ema_model.state_dict(), 'acc1': best_ema_acc}, "best_ema.pth")

        current_lr = optimizer.param_groups[0]['lr']
        print(
            f"Epoch {epoch:03d}/{epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
            f"ema_val_loss={ema_val_loss:.4f} ema_val_acc={ema_val_acc:.4f} | "
            f"best_acc={best_acc:.4f} best_ema_acc={best_ema_acc:.4f} | "
            f"lr={current_lr:.6f}"
        )


# ---------------------------
# Invoke main
# ---------------------------
if __name__ == "__main__":
    main()
