import torch
import torch.nn as nn
import torchvision.models as models


class AIFaceDetector(nn.Module):
    def __init__(self, freeze_early=True):
        """
        freeze_early: if True, freeze first 10 feature layers
                      keeps basic edge/texture detectors intact
                      only trains high-level feature layers + classifier
        """
        super(AIFaceDetector, self).__init__()

        # Load pretrained MobileNetV3
        self.model = models.mobilenet_v3_large(weights="IMAGENET1K_V1")

        # Replace classifier head — 1 output for binary classification
        self.model.classifier[3] = nn.Linear(1280, 1)

        # Freeze early layers if specified
        if freeze_early:
            self._freeze_early_layers()

    def _freeze_early_layers(self):
        """Freeze first 10 feature extraction layers.
        These detect basic edges and textures — already well trained.
        Layers 10-16 detect high level features — we retrain these.
        """
        for i, layer in enumerate(self.model.features):
            if i < 5:
                for param in layer.parameters():
                    param.requires_grad = False

    def forward(self, x):
        """Forward pass — returns raw logit (not sigmoid).
        Sigmoid applied during loss calculation, not here.
        """
        return self.model(x)

    def count_parameters(self):
        """Count trainable vs total parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


# Test the model
if __name__ == "__main__":
    model = AIFaceDetector(freeze_early=True)
    model.eval()

    # Check parameters
    total, trainable = model.count_parameters()
    print(f"Total parameters:     {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    print(f"Frozen parameters:    {total - trainable:,}")

    # Test forward pass
    dummy = torch.randn(1, 3, 224, 224)
    output = model(dummy)
    prob = torch.sigmoid(output)

    print(f"\nOutput shape: {output.shape}")
    print(f"Raw logit:    {output.item():.4f}")
    print(f"Probability:  {prob.item():.4f}")

    if prob.item() > 0.5:
        print("Prediction:   AI Generated")
    else:
        print("Prediction:   Real")