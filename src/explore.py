import os

splits = ['train', 'valid', 'test']
base_path = 'C:\\Users\\abhis\\Desktop\\ai_face_detector\\data\\real-vs-fake'

for split in splits:
    for label in ['real', 'fake']:
        path = os.path.join(base_path, split, label)
        count = len(os.listdir(path))
        print(f"{split}/{label}: {count} images")