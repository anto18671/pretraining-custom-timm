# ---------------------------
# Imports
# ---------------------------
import math

import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

import timm
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy, LabelSmoothingCrossEntropy
from timm.utils import ModelEmaV2
from tqdm import tqdm
import torchsummary


# ---------------------------
# Compose torchvision transforms with RandAugment and RandomErasing
# ---------------------------
def build_transforms(image_size: int):
    # Training transforms
    train_tfms = transforms.Compose([
        transforms.Resize(int(image_size * 1.15), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomResizedCrop(image_size, scale=(0.08, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3), value='random', inplace=False)
    ])

    # Validation transforms
    valid_tfms = transforms.Compose([
        transforms.Resize(int(image_size * 1.15), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    # Return all transforms
    return train_tfms, valid_tfms


# ---------------------------
# Create DataLoaders from imagenet1k/ with reproducible split
# ---------------------------
def create_dataloaders(image_size: int, batch_size: int, val_split: float, seed: int):
    # Build transforms
    train_tfms, valid_tfms = build_transforms(image_size)

    # Create dataset
    data_dir = "imagenet1k"
    full_ds = datasets.ImageFolder(root=data_dir, transform=train_tfms)

    # Split dataset
    class_to_idx = full_ds.class_to_idx
    num_items = len(full_ds)
    val_len = int(math.floor(num_items * val_split))
    train_len = num_items - val_len

    # Create split
    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(full_ds, [train_len, val_len], generator=generator)

    # Set validation transforms
    val_ds.dataset.transform = valid_tfms

    # Create loaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False, persistent_workers=True)

    # Return loaders and classes
    return train_loader, val_loader, class_to_idx


# ---------------------------
# Build model, loss, optimizer, EMA, device, mixup/cutmix
# ---------------------------
def build_training_objects(num_classes: int, base_lr: float, weight_decay: float, mixup_alpha: float, cutmix_alpha: float):
    # Select device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create model
    model = timm.create_model(
        "efficientvit_b3", 
        pretrained=False, 
        num_classes=num_classes, 
        drop_rate=0.25, 
        widths=(16, 32, 64, 128, 256), 
        depths=(1, 4, 6, 6, 9), 
        head_dim=16, 
        head_widths=(1024, 1080)
    )

    # Move and set memory format
    model = model.to(device).to(memory_format=torch.channels_last)

    # Build optimizer
    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay, betas=(0.9, 0.999), eps=1e-8)

    # Build EMA
    ema = ModelEmaV2(model, decay=0.9999)

    # Build losses
    train_criterion = SoftTargetCrossEntropy()
    eval_criterion = LabelSmoothingCrossEntropy(smoothing=0.1)

    # Build mixup
    mixup_fn = Mixup(
        mixup_alpha=mixup_alpha,
        cutmix_alpha=cutmix_alpha,
        prob=1.0,
        switch_prob=0.0,
        mode='batch',
        label_smoothing=0.0,
        num_classes=num_classes
    )

    # Return objects
    return model, optimizer, ema, train_criterion, eval_criterion, device, mixup_fn


# ---------------------------
# Warmup-cosine scheduler builder
# ---------------------------
def build_warmup_cosine_scheduler(optimizer: optim.Optimizer, warmup_epochs: int, total_epochs: int, steps_per_epoch: int):
    # Define lr lambda
    def lr_lambda(step: int):
        total_steps = total_epochs * steps_per_epoch
        warmup_steps = warmup_epochs * steps_per_epoch
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    # Build scheduler
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    # Return scheduler
    return scheduler


# ---------------------------
# Train for one epoch with AugReg and AMP
# ---------------------------
def train_one_epoch(model, ema, loader, train_criterion, optimizer, scheduler, device, scaler, mixup_fn, epoch, epochs, steps_done):
    # Set train mode
    model.train()

    # Initialize meters
    running_loss = 0.0
    running_correct = 0
    running_count = 0

    # Progress bar
    loop = tqdm(loader, desc=f"Train Epoch {epoch}/{epochs}", leave=False)

    # Iterate over batches
    for images, targets in loop:
        # Move inputs
        images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
        targets = targets.to(device, non_blocking=True)

        # Apply mixup
        images, targets = mixup_fn(images, targets)

        # Zero grad
        optimizer.zero_grad(set_to_none=True)

        # Forward and loss
        with torch.amp.autocast(device_type='cuda', enabled=device.type == 'cuda'):
            outputs = model(images)
            loss = train_criterion(outputs, targets)

        # Backward and step
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        # Scheduler step
        scheduler.step()
        steps_done += 1

        # EMA update
        ema.update(model)

        # Compute proxy acc
        preds = outputs.float().softmax(dim=1).argmax(dim=1)
        hard_targets = targets.float().argmax(dim=1)

        # Update meters
        running_loss += loss.item() * images.size(0)
        running_correct += (preds == hard_targets).sum().item()
        running_count += images.size(0)

        # Update progress
        avg_loss = running_loss / max(1, running_count)
        avg_acc = running_correct / max(1, running_count)
        loop.set_postfix(loss=avg_loss, acc=avg_acc)

    # Compute epoch metrics
    epoch_loss = running_loss / max(1, running_count)
    epoch_acc = running_correct / max(1, running_count)

    # Return metrics and step count
    return epoch_loss, epoch_acc, steps_done


# ---------------------------
# Evaluate on validation set with EMA
# ---------------------------
def evaluate(model_or_ema, loader, eval_criterion, device, epoch, epochs):
    # Set eval mode
    model_or_ema.eval()

    # Initialize meters
    running_loss = 0.0
    running_correct = 0
    running_count = 0

    # Progress bar
    loop = tqdm(loader, desc=f"Valid Epoch {epoch}/{epochs}", leave=False)

    # Disable grads
    with torch.no_grad():
        for images, targets in loop:
            # Move inputs
            images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
            targets = targets.to(device, non_blocking=True)

            # Forward
            outputs = model_or_ema(images)
            loss = eval_criterion(outputs, targets)

            # Predictions
            preds = outputs.argmax(dim=1)

            # Update meters
            running_loss += loss.item() * images.size(0)
            running_correct += (preds == targets).sum().item()
            running_count += images.size(0)

            # Update progress
            avg_loss = running_loss / max(1, running_count)
            avg_acc = running_correct / max(1, running_count)
            loop.set_postfix(loss=avg_loss, acc=avg_acc)

    # Compute metrics
    epoch_loss = running_loss / max(1, running_count)
    epoch_acc = running_correct / max(1, running_count)

    # Return metrics
    return epoch_loss, epoch_acc


# ---------------------------
# Training orchestration for pretraining
# ---------------------------
def main():
    # Repro and backend
    cudnn.benchmark = True

    # Config
    image_size = 224
    batch_size = 128
    val_split = 0.1
    seed = 42
    epochs = 300
    warmup_epochs = 20
    base_lr = 5e-4 * (batch_size / 512.0)
    weight_decay = 0.05
    mixup_alpha = 0.8
    cutmix_alpha = 1.0

    # Data
    train_loader, val_loader, class_to_idx = create_dataloaders(image_size=image_size, batch_size=batch_size, val_split=val_split, seed=seed)
    num_classes = len(class_to_idx)

    # Model and training objects
    model, optimizer, ema, train_criterion, eval_criterion, device, mixup_fn = build_training_objects(
        num_classes=num_classes,
        base_lr=base_lr,
        weight_decay=weight_decay,
        mixup_alpha=mixup_alpha,
        cutmix_alpha=cutmix_alpha
    )

    # AMP scaler
    scaler = torch.amp.GradScaler('cuda')

    # Summary
    torchsummary.summary(model, (3, image_size, image_size))

    # Scheduler
    steps_per_epoch = len(train_loader)
    scheduler = build_warmup_cosine_scheduler(optimizer, warmup_epochs=warmup_epochs, total_epochs=epochs, steps_per_epoch=steps_per_epoch)

    # Tracking
    best_acc = 0.0
    global_steps = 0

    # Epoch loop
    for epoch in range(1, epochs + 1):
        # Train
        train_loss, train_acc, global_steps = train_one_epoch(
            model=model,
            ema=ema,
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

        # Evaluate with EMA
        ema_model = ema.module
        val_loss, val_acc = evaluate(
            model_or_ema=ema_model,
            loader=val_loader,
            eval_criterion=eval_criterion,
            device=device,
            epoch=epoch,
            epochs=epochs
        )

        # Save checkpoints
        torch.save({'epoch': epoch, 'state_dict': model.state_dict()}, "last.pth")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({'epoch': epoch, 'state_dict': ema_model.state_dict(), 'acc1': best_acc}, "best_ema.pth")

        # Print metrics
        current_lr = optimizer.param_groups[0]['lr']
        print(
            f"Epoch {epoch:03d}/{epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
            f"best_acc={best_acc:.4f} | lr={current_lr:.6f}"
        )


# ---------------------------
# Invoke main
# ---------------------------
if __name__ == "__main__":
    main()
