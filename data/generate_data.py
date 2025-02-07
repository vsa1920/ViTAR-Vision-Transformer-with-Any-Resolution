import torch

from torchvision.datasets import Imagenette
import torchvision.transforms as transforms
from tqdm import tqdm

import os

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

data_dir = "/scratch2/vaibhav/data/imagenette2-320"
try:
    train_data = Imagenette(root=data_dir, split='train', download=True)
except:
    train_data = Imagenette(root=data_dir, split='train', download=False)
test_val_data = Imagenette(root=data_dir, split='val', download=False)

test_data, val_data = torch.utils.data.random_split(test_val_data, [int(0.5 * len(test_val_data)) + len(test_val_data) % 2, int(0.5 * len(test_val_data))])


def save_binned_dataset(dataset, bins, output_dir):
    """
    Perform a single pass through the dataset to preprocess and save binned datasets.

    Args:
        dataset: A PyTorch dataset (e.g., ImageFolder).
        bins: List of resolution bins to categorize the images.
        output_dir: Path to save the dataset organized by bins.

    Returns:
        None. The function saves resized images and labels to disk.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Create directories for each bin
    bin_dirs = {bin_size: os.path.join(output_dir, f"bin_{bin_size}") for bin_size in bins}
    for bin_dir in bin_dirs.values():
        os.makedirs(bin_dir, exist_ok=True)

    # Function to find the closest bin
    def find_closest_bin(resolution, bins):
        return min(bins, key=lambda x: abs(x - resolution))

    # Single pass through the dataset
    for idx in tqdm(range(len(dataset))):
        image, label = dataset[idx]

        resolution = max(image.size[1], image.size[0])  # Width, Height for PIL.Image
        closest_bin = find_closest_bin(resolution, bins)

        # Resize the image to the bin's resolution
        transform_resize = transforms.Compose([
            transforms.Resize((closest_bin, closest_bin)),  # Resize to square
        ])
        image_resized = transform_resize(image)

        # Save the resized image
        bin_dir = bin_dirs[closest_bin]
        image_path = os.path.join(bin_dir, f"{idx}.png")
        image_rgb = image_resized.convert('RGB')
        image_rgb.save(image_path)

        # Save the label
        with open(os.path.join(bin_dir, "labels.txt"), "a") as label_file:
            label_file.write(f"{idx}.png {label}\n")

    print("Binning and saving completed.")

if __name__ == "__main__":
    bins = [224, 448, 512]
    output_dir = "/scratch2/vaibhav/data/imagenette2-320_binned"
    print("Saving binned training dataset...")
    save_binned_dataset(train_data, bins, output_dir + "/train")
    print("Saving binned validation dataset...")
    save_binned_dataset(val_data, bins, output_dir + "/val")
    print("Saving binned test dataset...")
    save_binned_dataset(test_data, bins, output_dir + "/test")
    print("All datasets saved.")