import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import io
import os
from sklearn.metrics import f1_score
from torchvision import transforms

from dataset import FaceDataset, transform
from model import AIFaceDetector


def apply_jpeg_compression(image_pil, quality):
    """
    Compress image to JPEG at given quality then decompress.
    Simulates what happens when image is shared on social media.
    quality: 1-95 (lower = more compressed = more quality loss)
    """
    buffer = io.BytesIO()
    image_pil.save(buffer, format='JPEG', quality=quality)
    buffer.seek(0)
    return Image.open(buffer).copy()


def apply_resize_degradation(image_pil, small_size):
    """
    Shrink image to small_size then resize back to original.
    Permanently loses information — simulates social media resizing.
    """
    original_size = image_pil.size
    small = image_pil.resize((small_size, small_size), Image.BILINEAR)
    return small.resize(original_size, Image.BILINEAR)


class DegradedDataset(Dataset):
    """
    Wraps FaceDataset but applies degradation before transform.
    degradation_fn: function that takes PIL image, returns PIL image
    """
    def __init__(self, base_dataset, degradation_fn, transform=None):
        self.base_dataset = base_dataset
        self.degradation_fn = degradation_fn
        self.transform = transform

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        # Load raw image path and label from base dataset
        image_path = self.base_dataset.image_paths[idx]
        label = self.base_dataset.labels[idx]

        # Open image, apply degradation, then transform
        image = Image.open(image_path).convert('RGB')
        image = self.degradation_fn(image)

        if self.transform:
            image = self.transform(image)

        return image, label


def evaluate_robustness(model, dataset, device, batch_size=32):
    """Run evaluation on degraded dataset, return F1 and accuracy."""
    loader = DataLoader(dataset, batch_size=batch_size,
                       shuffle=False, num_workers=0)
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images).squeeze()
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    f1 = f1_score(all_labels, all_preds)
    accuracy = (np.array(all_labels) == np.array(all_preds)).mean()
    return f1, accuracy


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load base test dataset (no degradation yet)
    base_path = 'data/real-vs-fake'
    base_test = FaceDataset('test', base_path, transform=None, limit=500)

    # Load model
    model = AIFaceDetector(freeze_early=True).to(device)
    model.load_state_dict(torch.load('models/best_model.pth',
                          map_location=device))
    print("Model loaded\n")

    results = []

    # 1. Clean baseline (no degradation)
    clean_dataset = DegradedDataset(base_test, lambda x: x, transform=transform)
    f1, acc = evaluate_robustness(model, clean_dataset, device)
    results.append(('Clean (baseline)', f1, acc, 0.0))
    baseline_f1 = f1
    print(f"Clean baseline — F1: {f1:.4f} | Acc: {acc:.4f}")

    # 2. JPEG compression tests
    for quality in [75, 50, 25, 10]:
        degraded = DegradedDataset(
            base_test,
            lambda x, q=quality: apply_jpeg_compression(x, q),
            transform=transform
        )
        f1, acc = evaluate_robustness(model, degraded, device)
        drop = baseline_f1 - f1
        results.append((f'JPEG Quality {quality}', f1, acc, drop))
        print(f"JPEG Q{quality:2d} — F1: {f1:.4f} | Acc: {acc:.4f} | Drop: {drop:.4f}")

    # 3. Resize degradation tests
    for size in [128, 64, 32]:
        degraded = DegradedDataset(
            base_test,
            lambda x, s=size: apply_resize_degradation(x, s),
            transform=transform
        )
        f1, acc = evaluate_robustness(model, degraded, device)
        drop = baseline_f1 - f1
        results.append((f'Resize {size}px', f1, acc, drop))
        print(f"Resize {size}px — F1: {f1:.4f} | Acc: {acc:.4f} | Drop: {drop:.4f}")

    # Print final summary table
    print("\n" + "="*60)
    print(f"{'Degradation':<25} {'F1':>8} {'Accuracy':>10} {'Drop':>8}")
    print("="*60)
    for name, f1, acc, drop in results:
        drop_str = "—" if drop == 0 else f"-{drop:.4f}"
        print(f"{name:<25} {f1:>8.4f} {acc:>10.4f} {drop_str:>8}")
    print("="*60)