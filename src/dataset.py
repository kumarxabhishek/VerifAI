import torch
from torchvision import transforms
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
])


class FaceDataset(Dataset):
    def __init__(self, split, base_path, transform=None, limit=None):
        self.transform = transform
        self.image_paths = []
        self.labels = []

        for label_name, label_value in [('real', 0), ('fake', 1)]:
            folder = os.path.join(base_path, split, label_name)
            files = os.listdir(folder)
            if limit:
                files = files[:limit]
            for filename in files:
                if filename.endswith('.jpg') or filename.endswith('.png'):
                    self.image_paths.append(os.path.join(folder, filename))
                    self.labels.append(label_value)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


if __name__ == "__main__":
    base_path = 'C:\\Users\\abhis\\Desktop\\ai_face_detector\\data\\real-vs-fake'

    train_dataset = FaceDataset('train', base_path, transform=transform)
    valid_dataset = FaceDataset('valid', base_path, transform=transform)
    test_dataset = FaceDataset('test', base_path, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    print(f"Train size: {len(train_dataset)}")
    print(f"Valid size: {len(valid_dataset)}")
    print(f"Test size:  {len(test_dataset)}")

    images, labels = next(iter(train_loader))
    print(f"Batch image shape: {images.shape}")
    print(f"Batch label shape: {labels.shape}")
    print(f"Sample labels: {labels[:5]}")