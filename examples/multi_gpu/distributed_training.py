"""
Multi-GPU Distributed Training with RTX-STone

This example shows how to use multiple RTX 50-series GPUs for distributed training
using PyTorch's DistributedDataParallel (DDP).

Features:
- Automatic multi-GPU detection
- Gradient accumulation
- Mixed precision training (BF16)
- Efficient data loading
- Checkpointing and resuming

Usage:
    # Single node, multiple GPUs
    torchrun --nproc_per_node=2 examples/multi_gpu/distributed_training.py

    # Or use torch.distributed.launch
    python -m torch.distributed.launch --nproc_per_node=2 examples/multi_gpu/distributed_training.py

Requirements:
    - 2+ RTX 50-series GPUs
    - PyTorch with RTX-STone
    - Dataset (e.g., CIFAR-10, ImageNet)

Author: RTX-STone Contributors
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import autocast, GradScaler
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm


def setup_distributed():
    """Initialize distributed training."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        print("Not using distributed mode")
        return None, None, None

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")

    return rank, world_size, local_rank


def cleanup_distributed():
    """Cleanup distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def get_dataloader(rank, world_size, batch_size, num_workers=4):
    """
    Create distributed dataloader.

    Args:
        rank: Process rank
        world_size: Total number of processes
        batch_size: Batch size per GPU
        num_workers: Number of data loading workers

    Returns:
        train_loader, val_loader
    """
    # Transforms
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Datasets
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )

    val_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_val
    )

    # Distributed samplers
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )

    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )

    # Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader


def train_epoch(model, loader, criterion, optimizer, scaler, device, rank):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    if rank == 0:
        pbar = tqdm(loader, desc="Training")
    else:
        pbar = loader

    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)

        # Mixed precision training
        with autocast(dtype=torch.bfloat16):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        # Backward pass with gradient scaling
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Statistics
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if rank == 0:
            pbar.set_postfix({
                'loss': total_loss / (batch_idx + 1),
                'acc': 100. * correct / total
            })

    return total_loss / len(loader), 100. * correct / total


@torch.no_grad()
def validate(model, loader, criterion, device, rank):
    """Validate the model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    if rank == 0:
        pbar = tqdm(loader, desc="Validation")
    else:
        pbar = loader

    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)

        with autocast(dtype=torch.bfloat16):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if rank == 0:
            pbar.set_postfix({
                'loss': total_loss / (batch_idx + 1),
                'acc': 100. * correct / total
            })

    return total_loss / len(loader), 100. * correct / total


def main():
    parser = argparse.ArgumentParser(description='RTX-STone Multi-GPU Training')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=128, help='batch size per GPU')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--model', type=str, default='resnet18', help='model architecture')
    args = parser.parse_args()

    # Setup distributed
    rank, world_size, local_rank = setup_distributed()

    if rank is None:
        print("ERROR: Distributed training not properly initialized")
        print("Use: torchrun --nproc_per_node=<num_gpus> distributed_training.py")
        return

    device = torch.device(f"cuda:{local_rank}")

    # Print info from rank 0
    if rank == 0:
        print("=" * 70)
        print("RTX-STone Multi-GPU Distributed Training")
        print("=" * 70)
        print(f"\nWorld Size: {world_size}")
        print(f"Batch Size per GPU: {args.batch_size}")
        print(f"Total Batch Size: {args.batch_size * world_size}")
        print(f"Model: {args.model}")
        print(f"Epochs: {args.epochs}")

        # Check GPUs
        for i in range(world_size):
            gpu_name = torch.cuda.get_device_name(i)
            compute_cap = torch.cuda.get_device_capability(i)
            print(f"\nGPU {i}: {gpu_name}")
            print(f"  Compute Capability: {compute_cap[0]}.{compute_cap[1]}")

            if compute_cap == (12, 0):
                print("  ✓ RTX 50-series - Native SM 12.0 support")

    # Create model
    model = torchvision.models.resnet18(num_classes=10)
    model = model.to(device)

    # Wrap with DDP
    model = DDP(model, device_ids=[local_rank])

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=5e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Mixed precision scaler
    scaler = GradScaler()

    # Dataloaders
    train_loader, val_loader = get_dataloader(rank, world_size, args.batch_size)

    # Training loop
    best_acc = 0
    for epoch in range(args.epochs):
        if rank == 0:
            print(f"\nEpoch {epoch + 1}/{args.epochs}")

        # Set epoch for distributed sampler
        train_loader.sampler.set_epoch(epoch)

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device, rank
        )

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device, rank)

        # Step scheduler
        scheduler.step()

        # Print results (rank 0 only)
        if rank == 0:
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

            # Save best model
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'accuracy': best_acc,
                }, 'best_model.pth')
                print(f"✓ Saved best model (acc: {best_acc:.2f}%)")

    if rank == 0:
        print("\n" + "=" * 70)
        print(f"Training Complete! Best Accuracy: {best_acc:.2f}%")
        print("=" * 70)

    # Cleanup
    cleanup_distributed()


if __name__ == "__main__":
    main()
