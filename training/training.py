import torch

from torch.utils.data import DataLoader
from data.reload_data import ReloadedBinnedDataset
import torchvision.transforms as transforms
from tqdm import tqdm
from ViTAR import ViTAR
import os
import random

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

data_dir = "/scratch2/vaibhav/data/imagenette2-320"
train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)])

test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)])

# Function to create a dictionary of DataLoaders for all bins
def create_binned_dataloaders(output_dir, batch_size=32, shuffle=True):
    bin_dirs = [os.path.join(output_dir, d) for d in os.listdir(output_dir) if d.startswith("bin_")]
    binned_dataloaders = {}

    for bin_dir in bin_dirs:
        dataset = ReloadedBinnedDataset(bin_dir)
        bin_name = bin_dir.split("/")[-1].split("_")[1]
        binned_dataloaders[bin_name] = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return binned_dataloaders

def train_multi_resolution(input_data_dir, model, criterion, optimizer, device, num_epochs=10, batch_size=32):
    """
    Trains a model using multi-resolution datasets by randomly sampling from multiple dataloaders
    until all are exhausted in each epoch.

    Parameters:
        model (torch.nn.Module): The neural network model.
        dataloaders (list): List of PyTorch dataloaders with different image sizes.
        criterion (torch.nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer for model training.
        device (torch.device): Device ("cuda" or "cpu").
        num_epochs (int): Number of epochs to train.
    """
    model.to(device)
    dataloaders = create_binned_dataloaders(input_data_dir, batch_size=batch_size)
    for epoch in range(num_epochs):
        print(f"Epoch [{epoch+1}/{num_epochs}] starting...")

        # Create fresh iterators for all dataloaders
        iterators = [iter(dl) for dl in dataloaders]
        active_indices = list(range(len(dataloaders)))  # Track available dataloaders
        step_counter = 0
        running_loss = {key: 0.0 for key in dataloaders.keys()}
        total = {key: 0 for key in dataloaders.keys()}
        active_dataloaders = list(dataloaders.keys())

        while active_indices:
            # Randomly pick from active dataloaders
            idx = random.choice(active_indices)
            selected_iterator = iterators[idx]

            try:
                batch = next(selected_iterator)  # Get next batch
            except StopIteration:
                # Remove exhausted dataloader
                active_indices.remove(idx)
                continue  # Skip iteration if exhausted

            # Unpack batch
            images, labels = batch
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss[active_dataloaders[idx]] += loss.item()
            total[active_dataloaders[idx]] += 1

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step_counter += 1
            if step_counter % 100 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Dataloader: {idx}, Remaining: {len(active_indices)}, Loss: {loss.item():.4f}")
            print(f"Epoch [{epoch+1}/{num_epochs}], Dataloader: {idx}, Remaining: {len(active_indices)}, Loss: {loss.item():.4f}")

        print(f"Epoch [{epoch+1}/{num_epochs}] completed.\n Average Loss: {running_loss/total}")


if __name__ == '__main__':
    model = ViTAR()
    data_dir = "/scratch2/vaibhav/data/imagenette2-320_binned"
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(data_dir + '/train', model.parameters(), lr=1e-5, weight_decay=0.0001)
    train_multi_resolution(model, criterion, optimizer)