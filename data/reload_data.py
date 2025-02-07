from torch.utils.data import Dataset
import os
from PIL import Image

class ReloadedBinnedDataset(Dataset):
    def __init__(self, bin_dir, transforms=None):
        """
        Args:
            bin_dir: Path to the bin directory containing images and labels.txt.
        """
        self.image_dir = bin_dir
        self.labels_path = os.path.join(bin_dir, "labels.txt")
        self.data = []

        with open(self.labels_path, "r") as f:
            for line in f:
                image_name, label = line.strip().split()
                self.data.append((image_name, int(label)))
        
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_name, label = self.data[idx]
        image_path = os.path.join(self.image_dir, image_name)
        image = Image.open(image_path).convert("RGB")

        if self.transforms is not None:
            image = self.transforms(image)

        return image, label
