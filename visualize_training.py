import pickle
import numpy as np
import matplotlib.pyplot as plt

with open("/scratch2/vaibhav/models/vitar/train_losses.pkl", "rb") as f:
    train_losses = pickle.load(f)

with open("/scratch2/vaibhav/models/vitar/val_losses.pkl", "rb") as f:
    val_losses = pickle.load(f)

with open("/scratch2/vaibhav/models/vitar/train_accuracies.pkl", "rb") as f:
    train_accuracies = pickle.load(f)

with open("/scratch2/vaibhav/models/vitar/val_accuracies.pkl", "rb") as f:
    val_accuracies = pickle.load(f)


for key in train_losses.keys():
    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(1, len(train_losses[key]) + 1), train_losses[key], label="Train Loss")
    plt.plot(np.arange(1, len(val_losses[key]) + 1), val_losses[key], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss vs Epoch for {key}")
    plt.legend()
    plt.savefig(f"/scratch2/vaibhav/models/vitar/plots/{key}_loss.png")

for key in train_accuracies.keys():
    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(1, len(train_accuracies[key]) + 1), train_accuracies[key], label="Train Accuracy")
    plt.plot(np.arange(1, len(val_accuracies[key]) + 1), val_accuracies[key], label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy vs Epoch for {key}")
    plt.legend()
    plt.savefig(f"/scratch2/vaibhav/models/vitar/plots/{key}_accuracy.png")