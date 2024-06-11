import json
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class CustomDataset(Dataset):
    def __init__(self, json_path, split):
        self.data = self.load_data(json_path, split)
        self.split = split
        

    def load_data(self, json_path, split):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data[split]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = item['image']
        image = Image.open(image_path)
        text = item['text']
        return {"image": image, "text": text}
