"""Vanilla CE training on GPU for CIFAR-10/100 with ResNet-18 (32x32).

Reproduces OpenOOD's baseline training exactly:
- ResNet-18 with 3x3 conv1, no maxpool (32x32 input)
- SGD: lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True
- Cosine annealing per step, lr_min=1e-6
- batch_size=128
- Standard CIFAR augmentation: RandomCrop(32, padding=4), RandomHorizontalFlip

Saves checkpoint compatible with OpenOOD evaluation.
"""

import argparse
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms


# ---- ResNet-18 for 32x32 (identical to OpenOOD) ----

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class ResNet18_32x32(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)
        self.feature_size = 512

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, x, return_feature=False, return_feature_list=False):
        f1 = F.relu(self.bn1(self.conv1(x)))
        f2 = self.layer1(f1)
        f3 = self.layer2(f2)
        f4 = self.layer3(f3)
        f5 = self.layer4(f4)
        f5 = self.avgpool(f5)
        feature = f5.view(f5.size(0), -1)
        logits = self.fc(feature)
        if return_feature:
            return logits, feature
        elif return_feature_list:
            return logits, [f1, f2, f3, f4, f5]
        return logits

    def forward_threshold(self, x, threshold):
        f1 = F.relu(self.bn1(self.conv1(x)))
        f2 = self.layer1(f1)
        f3 = self.layer2(f2)
        f4 = self.layer3(f3)
        f5 = self.layer4(f4)
        f5 = self.avgpool(f5)
        feature = f5.clip(max=threshold)
        feature = feature.view(feature.size(0), -1)
        return self.fc(feature)

    def get_fc(self):
        fc = self.fc
        return fc.weight.cpu().detach().numpy(), fc.bias.cpu().detach().numpy()

    def get_fc_layer(self):
        return self.fc


# ---- Cosine annealing (identical to OpenOOD) ----

def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (1 + math.cos(step / total_steps * math.pi))


# ---- Data ----

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)


def get_dataloaders(dataset_name, data_dir, batch_size=128):
    if dataset_name == 'cifar10':
        mean, std = CIFAR10_MEAN, CIFAR10_STD
        Dataset = torchvision.datasets.CIFAR10
        num_classes = 10
    else:
        mean, std = CIFAR100_MEAN, CIFAR100_STD
        Dataset = torchvision.datasets.CIFAR100
        num_classes = 100

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_set = Dataset(data_dir, train=True, download=True, transform=train_transform)
    test_set = Dataset(data_dir, train=False, download=True, transform=test_transform)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=4, drop_last=False, pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=200, shuffle=False,
        num_workers=4, pin_memory=True,
    )
    return train_loader, test_loader, num_classes


# ---- Training ----

def train(args):
    device = torch.device('cuda')
    print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")

    # Set seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    train_loader, test_loader, num_classes = get_dataloaders(
        args.dataset, args.data_dir, args.batch_size
    )

    model = ResNet18_32x32(num_classes=num_classes).to(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True,
    )

    total_steps = args.num_epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: cosine_annealing(step, total_steps, 1, 1e-6 / args.lr),
    )

    criterion = nn.CrossEntropyLoss()

    # Output directory (matches OpenOOD naming)
    exp_name = f"{args.dataset}_resnet18_32x32_base_e{args.num_epochs}_lr{args.lr}_default"
    out_dir = os.path.join(args.output_dir, exp_name, f"s{args.seed}")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Training {args.dataset} for {args.num_epochs} epochs, seed {args.seed}")
    print(f"Output: {out_dir}")
    print(f"Total steps: {total_steps}")

    best_acc = 0.0

    for epoch in range(args.num_epochs):
        model.train()
        loss_sum = 0.0
        correct = 0
        total = 0
        t0 = time.time()

        for inputs, targets in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            scheduler.step()

            loss_sum += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        train_loss = loss_sum / total
        train_acc = 100.0 * correct / total
        elapsed = time.time() - t0

        # Evaluate every 10 epochs or at the end
        if (epoch + 1) % 10 == 0 or (epoch + 1) == args.num_epochs:
            test_acc = evaluate(model, test_loader, device)
            print(f"Epoch {epoch+1}/{args.num_epochs}  "
                  f"train_loss={train_loss:.4f}  train_acc={train_acc:.2f}%  "
                  f"test_acc={test_acc:.2f}%  lr={scheduler.get_last_lr()[0]:.6f}  "
                  f"time={elapsed:.1f}s")

            if test_acc > best_acc:
                best_acc = test_acc
                save_checkpoint(model, out_dir, "best.ckpt")
        else:
            print(f"Epoch {epoch+1}/{args.num_epochs}  "
                  f"train_loss={train_loss:.4f}  train_acc={train_acc:.2f}%  "
                  f"lr={scheduler.get_last_lr()[0]:.6f}  time={elapsed:.1f}s")

    # Save final checkpoint
    save_checkpoint(model, out_dir, "best_epoch.ckpt")
    save_checkpoint(model, out_dir, "last_epoch.ckpt")

    final_test_acc = evaluate(model, test_loader, device)
    print(f"\nFinal test accuracy: {final_test_acc:.2f}%")
    print(f"Best test accuracy: {best_acc:.2f}%")
    print(f"Checkpoint saved to {out_dir}")


def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return 100.0 * correct / total


def save_checkpoint(model, out_dir, filename):
    """Save in OpenOOD-compatible format (state dict on CPU)."""
    cpu_state = {k: v.cpu() for k, v in model.state_dict().items()}
    path = os.path.join(out_dir, filename)
    torch.save({"net": cpu_state}, path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100"])
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--output-dir", type=str, default="./results")
    args = parser.parse_args()
    train(args)
