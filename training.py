import torch

from torch.utils.data import DataLoader
from data import ReloadedBinnedDataset
import torchvision.transforms as transforms
from model import ViTAR
import os
import random

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)])

test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)])

# Function to create a dictionary of DataLoaders for all bins
def create_binned_dataloaders(output_dir, batch_size=32, shuffle=True, data_transforms=None):
    bin_dirs = [os.path.join(output_dir, d) for d in os.listdir(output_dir) if d.startswith("bin_")]
    binned_dataloaders = {}

    for bin_dir in bin_dirs:
        bin_name = bin_dir.split("/")[-1].split("_")[1]
        if bin_name != "512":
            continue
        dataset = ReloadedBinnedDataset(bin_dir, transforms=data_transforms)
        binned_dataloaders[bin_name] = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return binned_dataloaders

def train_multi_resolution(data_dir, model, criterion, optimizer, device, scheduler=None, num_epochs=10, batch_size=32):
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
    # train loaders
    train_path = data_dir + "/train"
    val_path = data_dir + "/val"
    dataloaders = create_binned_dataloaders(train_path, batch_size=batch_size, data_transforms=train_transforms)
    val_dataloaders = create_binned_dataloaders(val_path, batch_size=batch_size, data_transforms=test_transforms)
    train_losses = {key: [] for key in dataloaders.keys()}
    val_losses = {key: [] for key in dataloaders.keys()}
    train_accuracies = {key: [] for key in dataloaders.keys()}
    val_accuracies = {key: [] for key in dataloaders.keys()}
    for epoch in range(num_epochs):
        print(f"Epoch [{epoch+1}/{num_epochs}] starting...")
        model.train()
        # Create fresh iterators for all dataloaders
        iterators = [iter(dataloaders[dl]) for dl in dataloaders]
        active_indices = list(range(len(dataloaders)))  # Track available dataloaders
        step_counter = 0
        running_loss = {key: 0.0 for key in dataloaders.keys()}
        total = {key: 0 for key in dataloaders.keys()}
        active_dataloaders = list(dataloaders.keys())
        correct = {key: 0 for key in dataloaders.keys()}
        val_correct = {key: 0 for key in dataloaders.keys()}

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
            total[active_dataloaders[idx]] += labels.size(0)
            
            correct[active_dataloaders[idx]] += (torch.argmax(outputs, dim=1) == labels).sum().item()
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step_counter += 1
            if step_counter % 50 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Remaining: {len(active_indices)}, Loss: {loss.item():.4f}")
            
            # Test run break
            # if step_counter > 20:
            #     break

        print(f"Epoch [{epoch+1}/{num_epochs}] completed.\n")
        val_running_loss = {key: 0.0 for key in dataloaders.keys()}
        val_total = {key: 0 for key in dataloaders.keys()}

        model.eval()
        for key in dataloaders.keys():
            for image, label in val_dataloaders[key]:
                image, label = image.to(device), label.to(device)
                output = model(image)
                loss = criterion(output, label)
                val_running_loss[key] += loss.item()
                val_total[key] += label.size(0)
                val_correct[key] += (torch.argmax(output, dim=1) == label).sum().item()
        
        if scheduler is not None:
            scheduler.step()

        for key in running_loss:
            print('\n')
            print(f"Average training loss for {key}: {running_loss[key]/total[key]:.4f}")
            print(f"Training accuracy for {key}: {correct[key]/total[key]:.4f}")
            print(f"Average validation loss for {key}: {val_running_loss[key]/val_total[key]:.4f}")
            print(f"Validation accuracy for {key}: {val_correct[key]/val_total[key]:.4f}")
            train_losses[key].append(running_loss[key]/total[key])
            val_losses[key].append(val_running_loss[key]/val_total[key])
            train_accuracies[key].append(correct[key]/total[key])
            val_accuracies[key].append(val_correct[key]/val_total[key])
        
    return train_losses, val_losses, train_accuracies, val_accuracies


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ViTAR()
    data_dir = "/scratch2/vaibhav/data/imagenette2-320_binned"
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    train_losses, val_losses, train_accuracies, val_accuracies = train_multi_resolution(data_dir, model, criterion, optimizer, device=device, scheduler=scheduler, num_epochs=30, batch_size=64)
    torch.save(model.state_dict(), "/scratch2/vaibhav/models/vitar/model.pth")
    print("Model saved successfully.")
    import pickle
    with open("/scratch2/vaibhav/models/vitar/train_losses.pkl", "wb") as f:
        pickle.dump(train_losses, f)
    with open("/scratch2/vaibhav/models/vitar/val_losses.pkl", "wb") as f:
        pickle.dump(val_losses, f)
    with open("/scratch2/vaibhav/models/vitar/train_accuracies.pkl", "wb") as f:
        pickle.dump(train_accuracies, f)
    with open("/scratch2/vaibhav/models/vitar/val_accuracies.pkl", "wb") as f:
        pickle.dump(val_accuracies, f)