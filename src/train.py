import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time

from dataset import FaceDataset, transform
from model import AIFaceDetector


def train_one_epoch(model, train_loader, optimizer, criterion, device):
    """Run one full epoch of training."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.float().to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(images).squeeze()
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Track metrics
        total_loss += loss.item()
        predictions = (torch.sigmoid(outputs) > 0.5).float()
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

        if batch_idx % 200 == 0:
            print(f"  Batch {batch_idx}/{len(train_loader)} — Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total
    return avg_loss, accuracy


def validate(model, valid_loader, criterion, device):
    """Run validation, no weight updates."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in valid_loader:
            images = images.to(device)
            labels = labels.float().to(device)

            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(valid_loader)
    accuracy = correct / total
    return avg_loss, accuracy


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Hyperparameters
    BATCH_SIZE = 32
    LEARNING_RATE = 0.0001
    EPOCHS = 5

    # Data — using subset for CPU training
    base_path = 'data/real-vs-fake'
    train_dataset = FaceDataset('train', base_path, transform=transform, limit=50000)
    valid_dataset = FaceDataset('valid', base_path, transform=transform, limit=10000)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Valid samples: {len(valid_dataset)}")

    # Model
    model = AIFaceDetector(freeze_early=True).to(device)

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    best_valid_loss = float('inf')

    for epoch in range(EPOCHS):
        start_time = time.time()
        print(f"\nEpoch {epoch+1}/{EPOCHS}")

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        valid_loss, valid_acc = validate(model, valid_loader, criterion, device)

        elapsed = time.time() - start_time

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Valid Loss: {valid_loss:.4f} | Valid Acc: {valid_acc:.4f}")
        print(f"Time: {elapsed:.1f}s")

        # Save best model
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'models/best_model.pth')
            print("Saved new best model")

    print("\nTraining complete!")