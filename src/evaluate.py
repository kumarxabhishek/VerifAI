import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    f1_score,
    confusion_matrix,
    precision_score,
    recall_score,
    classification_report
)
import numpy as np
import os

from dataset import FaceDataset, transform
from model import AIFaceDetector


def evaluate(model, test_loader, device, threshold=0.5):
    """
    Run full evaluation on test set.
    Returns all predictions and true labels.
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images).squeeze()

            # Convert logits to probabilities
            probs = torch.sigmoid(outputs)

            # Apply threshold
            preds = (probs > threshold).float()

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    return (
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs)
    )


def find_best_threshold(labels, probs):
    """
    Try multiple thresholds, find one with best F1.
    Most people skip this — don't.
    """
    best_f1 = 0
    best_threshold = 0.5

    for threshold in np.arange(0.1, 0.9, 0.05):
        preds = (probs > threshold).astype(float)
        f1 = f1_score(labels, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold, best_f1


def print_results(labels, preds, probs):
    """Print all evaluation metrics clearly."""

    f1 = f1_score(labels, preds)
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    accuracy = (labels == preds).mean()
    cm = confusion_matrix(labels, preds)

    print("=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")

    print("\nConfusion Matrix:")
    print("                Pred Real  Pred AI")
    print(f"Actual Real  →    {cm[0][0]:5d}    {cm[0][1]:5d}")
    print(f"Actual AI    →    {cm[1][0]:5d}    {cm[1][1]:5d}")

    print("\nDetailed Report:")
    print(classification_report(labels, preds,
          target_names=['Real', 'AI Generated']))

    # Find best threshold
    best_thresh, best_f1 = find_best_threshold(labels, probs)
    print(f"Best threshold: {best_thresh:.2f} → F1: {best_f1:.4f}")
    if best_thresh != 0.5:
        print(f"Note: default 0.5 threshold is suboptimal for this model")

    print("=" * 50)


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load test dataset
    base_path = 'data/real-vs-fake'
    test_dataset = FaceDataset('test', base_path, transform=transform, limit=5000)
    test_loader = DataLoader(test_dataset, batch_size=32,
                            shuffle=False, num_workers=0)

    print(f"Test samples: {len(test_dataset)}")

    # Load trained model
    model = AIFaceDetector(freeze_early=True).to(device)
    model.load_state_dict(torch.load('models/best_model.pth',
                          map_location=device))
    print("Model loaded successfully")

    # Run evaluation
    labels, preds, probs = evaluate(model, test_loader, device)

    # Print results
    print_results(labels, preds, probs)