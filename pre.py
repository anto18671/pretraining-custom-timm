import glob
import math
import os
import sys
from typing import Tuple

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import timm
import torchsummary
from tqdm import tqdm
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy, LabelSmoothingCrossEntropy
from huggingface_hub import snapshot_download, login
from datasets import load_dataset


# ---------------------------
# Compose torchvision transforms with RandAugment and RandomErasing
# ---------------------------
def build_transforms(image_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
    # Build training transforms
    train_tfms = transforms.Compose([
        transforms.Resize(int(image_size * 1.15), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomResizedCrop(image_size, scale=(0.08, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3), value='random', inplace=False)
    ])

    # Build validation transforms
    valid_tfms = transforms.Compose([
        transforms.Resize(int(image_size * 1.15), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    return train_tfms, valid_tfms


# ---------------------------
# Thin PyTorch Dataset wrapper around HF parquet splits
# ---------------------------
class HFImageFolder(Dataset):
    # Initialize dataset with split and transform
    def __init__(self, hf_split, transform):
        self.data = hf_split
        self.transform = transform

    # Return length of dataset
    def __len__(self):
        return len(self.data)

    # Return one (image, label) pair
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
    # Define candidate names for splits
    train_candidates = ["train", "training"]
    val_candidates = ["val", "valid", "validation"]

    # Initialize discovered names
    train_name = None
    val_name = None

    # Search for training split
    for name in train_candidates:
        if os.path.isdir(os.path.join(local_dir, name)):
            train_name = name
            break

    # Search for validation split
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
# Download dataset repo from Hugging Face Hub
# ---------------------------
def download_dataset_repo(repo_id: str, revision: str, token_env: str, local_cache_dir: str) -> str:
    # Authenticate if token is present
    if token_env and len(token_env.strip()) > 0:
        login(token=token_env.strip())

    # Download snapshot locally
    local_dir = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision if revision else None,
        local_dir=local_cache_dir if local_cache_dir else None,
        local_dir_use_symlinks=False
    )

    return local_dir


# ---------------------------
# Ensure parquet shards exist locally, otherwise download them
# ---------------------------
def ensure_parquet_dataset(repo_id: str, revision: str, token_env: str, local_cache_dir: str) -> str:
    # Define expected parquet directory
    parquet_dir = os.path.join(local_cache_dir, "data")

    # Define glob patterns
    train_glob = os.path.join(parquet_dir, "train-*.parquet")
    val_glob = os.path.join(parquet_dir, "validation-*.parquet")

    # Check if files exist
    have_train = len(glob.glob(train_glob)) > 0
    have_val = len(glob.glob(val_glob)) > 0

    # If both are found, return directory
    if have_train and have_val:
        return parquet_dir

    # Otherwise, trigger download
    local_dir = download_dataset_repo(
        repo_id=repo_id,
        revision=revision,
        token_env=token_env,
        local_cache_dir=local_cache_dir
    )

    # Re-check after download
    parquet_dir = os.path.join(local_dir, "data")
    train_glob = os.path.join(parquet_dir, "train-*.parquet")
    val_glob = os.path.join(parquet_dir, "validation-*.parquet")
    have_train = len(glob.glob(train_glob)) > 0
    have_val = len(glob.glob(val_glob)) > 0

    # Fail if still missing
    if not (have_train and have_val):
        raise FileNotFoundError(
            f"Parquet shards not found under '{parquet_dir}'. "
            f"Expected train-*.parquet and validation-*.parquet."
        )

    return parquet_dir


# ---------------------------
# Create DataLoaders from local parquet shards
# ---------------------------
def create_dataloaders_from_parquet(image_size: int, batch_size: int, data_dir: str):
    # Build transforms
    train_tfms, valid_tfms = build_transforms(image_size)

    # Define glob patterns
    train_glob = os.path.join(data_dir, "train-*.parquet")
    val_glob = os.path.join(data_dir, "validation-*.parquet")

    # Load datasets
    ds = load_dataset(
        "parquet",
        data_files={
            "train": train_glob,
            "validation": val_glob
        }
    )

    # Wrap datasets
    train_ds = HFImageFolder(ds["train"], transform=train_tfms)
    val_ds = HFImageFolder(ds["validation"], transform=valid_tfms)

    # Derive class mapping
    try:
        label_feature = ds["train"].features["label"]
        class_to_idx = {name: i for i, name in enumerate(label_feature.names)}
    except Exception:
        try:
            max_label = max(int(x["label"]) for x in ds["train"])
            class_to_idx = {str(i): i for i in range(max_label + 1)}
        except Exception:
            class_to_idx = {}

    # Build DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=12,
        drop_last=True,
        persistent_workers=True,
        prefetch_factor=4
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        drop_last=True,
        persistent_workers=True,
        prefetch_factor=4
    )

    return train_loader, val_loader, class_to_idx


# ---------------------------
# Build model, loss, optimizer, device, and mixup
# ---------------------------
def build_training_objects(num_classes: int, base_lr: float, weight_decay: float, mixup_alpha: float, cutmix_alpha: float):
    # Select device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create EfficientViT model
    model = timm.create_model(
        "efficientvit_b3",
        pretrained=False,
        num_classes=num_classes,
        drop_rate=0.2,
        widths=(32, 64, 128, 256, 512),
        depths=(1, 2, 3, 3, 4),
        head_dim=24,
        head_widths=(2048, 1024)
    )

    # Move model to device
    model = model.to(device).to(memory_format=torch.channels_last)

    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=base_lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )

    # Define training loss
    train_criterion = SoftTargetCrossEntropy()

    # Define evaluation loss
    eval_criterion = LabelSmoothingCrossEntropy(smoothing=0.1)

    # Define mixup function
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
# Build EMA model using PyTorch AveragedModel
# ---------------------------
def build_ema_model(model: torch.nn.Module, ema_decay: float):
    # Define EMA update function
    def ema_avg_fn(averaged_param, current_param, num_averaged):
        return ema_decay * averaged_param + (1.0 - ema_decay) * current_param

    # Wrap model with AveragedModel
    ema_model = AveragedModel(model, avg_fn=ema_avg_fn, use_buffers=True)

    return ema_model


# ---------------------------
# Build warmup-cosine learning rate scheduler
# ---------------------------
def build_warmup_cosine_scheduler(optimizer: optim.Optimizer, warmup_epochs: int, total_epochs: int, steps_per_epoch: int):
    # Define learning rate schedule function
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
# Train for one epoch with AMP and EMA
# ---------------------------
def train_one_epoch(model, ema_model, loader, train_criterion, optimizer, scheduler, device, scaler, mixup_fn, epoch, epochs, steps_done, accumulation_steps):
    # Set model to training mode
    model.train()

    # Initialize running statistics
    running_loss = 0.0
    running_correct = 0
    running_count = 0

    # Initialize accumulation counter
    micro_step = 0

    # Create tqdm loop
    loop = tqdm(loader, desc=f"Train Epoch {epoch}/{epochs}", leave=False)

    # Iterate over mini-batches
    for images, targets in loop:
        # Move images and labels to device
        images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
        targets = targets.to(device, non_blocking=True)

        # Apply mixup augmentation
        images, targets = mixup_fn(images, targets)

        # Reset optimizer gradients at the start of an accumulation cycle
        if micro_step % accumulation_steps == 0:
            optimizer.zero_grad(set_to_none=True)

        # Forward pass with AMP
        with torch.amp.autocast('cuda'):
            outputs = model(images)
            full_loss = train_criterion(outputs, targets)

        # Scale loss by accumulation steps
        loss = full_loss / accumulation_steps

        # Backward pass with gradient scaling
        scaler.scale(loss).backward()

        # Perform optimizer step at accumulation boundary
        if (micro_step + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            # Update EMA parameters after optimizer step
            ema_model.update_parameters(model)

            # Step learning rate scheduler once per optimizer update
            scheduler.step()
            steps_done += 1

            # Reset gradients for next cycle
            optimizer.zero_grad(set_to_none=True)

        # Compute predictions and hard targets
        preds = outputs.float().softmax(dim=1).argmax(dim=1)
        hard_targets = targets.float().argmax(dim=1)

        # Update statistics using unscaled loss
        running_loss += full_loss.item() * images.size(0)
        running_correct += (preds == hard_targets).sum().item()
        running_count += images.size(0)

        # Compute averages
        avg_loss = running_loss / max(1, running_count)
        avg_acc = running_correct / max(1, running_count)

        # Update tqdm display
        loop.set_postfix(loss=avg_loss, acc=avg_acc)

        # Increment micro step
        micro_step += 1

    # Final averages
    epoch_loss = running_loss / max(1, running_count)
    epoch_acc = running_correct / max(1, running_count)

    return epoch_loss, epoch_acc, steps_done


# ---------------------------
# Evaluate model on validation set
# ---------------------------
def evaluate(model, loader, eval_criterion, device, epoch, epochs):
    # Set model to evaluation mode
    model.eval()

    # Initialize running statistics
    running_loss = 0.0
    running_correct = 0
    running_count = 0

    # Create tqdm loop
    loop = tqdm(loader, desc=f"Valid Epoch {epoch}/{epochs}", leave=False)

    # Disable gradients
    with torch.no_grad():
        # Iterate over validation mini-batches
        for images, targets in loop:
            # Move images and labels to device
            images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
            targets = targets.to(device, non_blocking=True)

            # Forward pass
            outputs = model(images)
            loss = eval_criterion(outputs, targets)

            # Compute predictions
            preds = outputs.argmax(dim=1)

            # Update statistics
            running_loss += loss.item() * images.size(0)
            running_correct += (preds == targets).sum().item()
            running_count += images.size(0)

            # Compute averages
            avg_loss = running_loss / max(1, running_count)
            avg_acc = running_correct / max(1, running_count)

            # Update tqdm display
            loop.set_postfix(loss=avg_loss, acc=avg_acc)

    # Final averages
    epoch_loss = running_loss / max(1, running_count)
    epoch_acc = running_correct / max(1, running_count)

    return epoch_loss, epoch_acc


# ---------------------------
# Training orchestration with EMA and Hub auto-download
# ---------------------------
def main():
    # Enable cudnn optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    # Define dataset repository information
    repo_id = os.environ.get("HF_IMAGENET_REPO", "benjamin-paine/imagenet-1k-256x256")
    revision = os.environ.get("HF_IMAGENET_REVISION", None)

    # Define authentication token
    token_env = os.environ.get("HUGGINGFACE_TOKEN", "")

    # Define local cache directory
    local_cache_dir = os.environ.get("HF_DATA_CACHE_DIR", "./imagenet1k")

    # Define training hyperparameters
    image_size = 256
    batch_size = 32
    epochs = 100
    warmup_epochs = 6
    base_lr = 5e-4 * (batch_size / 512.0)
    weight_decay = 0.05
    mixup_alpha = 0.8
    cutmix_alpha = 1.0
    ema_decay = 0.9999

    # Define gradient accumulation steps
    accumulation_steps = 8

    # Ensure parquet dataset is available locally
    try:
        parquet_dir = ensure_parquet_dataset(
            repo_id=repo_id,
            revision=revision,
            token_env=token_env,
            local_cache_dir=local_cache_dir
        )
    except Exception as e:
        print(f"[FATAL] Failed to ensure dataset. Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Create DataLoaders from parquet dataset
    try:
        train_loader, val_loader, class_to_idx = create_dataloaders_from_parquet(
            image_size=image_size,
            batch_size=batch_size,
            data_dir=parquet_dir
        )
    except Exception as e:
        print(f"[FATAL] Failed to build dataloaders from parquet in '{parquet_dir}'. Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Determine number of classes
    num_classes = len(class_to_idx) if len(class_to_idx) > 0 else 1000

    # Build model, optimizer, loss functions, and mixup
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

    # Create AMP gradient scaler
    scaler = torch.amp.GradScaler("cuda")

    # Print model summary
    torchsummary.summary(model, (3, image_size, image_size))

    # Compute effective optimizer steps per epoch
    steps_per_epoch = math.ceil(len(train_loader) / max(1, accumulation_steps))

    # Build learning rate scheduler
    scheduler = build_warmup_cosine_scheduler(
        optimizer,
        warmup_epochs=warmup_epochs,
        total_epochs=epochs,
        steps_per_epoch=steps_per_epoch
    )

    # Initialize best accuracy trackers
    best_acc = 0.0
    best_ema_acc = 0.0
    global_steps = 0

    # Train across epochs
    for epoch in range(1, epochs + 1):
        # Train for one epoch
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
            steps_done=global_steps,
            accumulation_steps=accumulation_steps
        )

        # Evaluate base model
        val_loss, val_acc = evaluate(
            model=model,
            loader=val_loader,
            eval_criterion=eval_criterion,
            device=device,
            epoch=epoch,
            epochs=epochs
        )

        # Evaluate EMA model
        ema_val_loss, ema_val_acc = evaluate(
            model=ema_model,
            loader=val_loader,
            eval_criterion=eval_criterion,
            device=device,
            epoch=epoch,
            epochs=epochs
        )

        # Save last checkpoint
        torch.save({'epoch': epoch, 'state_dict': model.state_dict()}, "last.pth")
        torch.save({'epoch': epoch, 'state_dict': ema_model.state_dict()}, "last_ema.pth")

        # Save best base model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'acc1': best_acc}, "best.pth")

        # Save best EMA model
        if ema_val_acc > best_ema_acc:
            best_ema_acc = ema_val_acc
            torch.save({'epoch': epoch, 'state_dict': ema_model.state_dict(), 'acc1': best_ema_acc}, "best_ema.pth")

        # Retrieve current learning rate
        current_lr = optimizer.param_groups[0]['lr']

        # Print epoch summary
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
