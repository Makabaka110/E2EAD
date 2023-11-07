import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
import csv
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
# Import the custom classes
from dataloader import *
from networks import *


# Define the file directory
features_directory = './data/'
labels_file = './data/driving_log.csv'
model_pth = './model.pth'
dataset_class = SteeringDatasetLSTM
model_class = SteeringModelLSTM

# # Load the data and transform to PyTorch tensors
# # Very important parameter, defining the shift variable for left and right steering angle
# delta = 0.2
# features, labels = data_loading(delta,labels_file,features_directory)

#load the features and labels from .npy
features = np.load('./features.npy')
labels = np.load('./labels.npy')

# Split the data into train and validation sets
dataset = dataset_class(features, labels, transform=transforms.ToTensor())
train_dataset, val_dataset = random_split(dataset, [int(len(dataset)*0.9), len(dataset)-int(len(dataset)*0.9)])

# Define the data loaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

model = model_class()
# Use CUDA if available
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)
# #print model datatype float or double
# print(model.resnet.fc.weight.dtype)

# Optimize
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0012, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0)

train_losses = []
val_losses = []

best_val_loss = np.inf
patience = 5
num_epochs_no_improvement = 0

print('Training started...')
for epoch in range(10):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        # print(inputs.dtype)
        # print(labels.dtype)
        inputs, labels = inputs.to(device).float(), labels.to(device).float()
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels.unsqueeze(1)).float()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    train_losses.append(running_loss / len(train_dataset))

    with torch.no_grad():
        val_loss = 0.0
        for i, data in enumerate(val_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1))
            val_loss += loss.item()
        val_losses.append(val_loss / len(val_dataset))

    print(f"Epoch {epoch+1} - Train Loss: {train_losses[-1]:.8f} - Val Loss: {val_losses[-1]:.8f}")

    # Check for early stopping
    if val_losses[-1] < best_val_loss:
        best_val_loss = val_losses[-1]
        num_epochs_no_improvement = 0
    else:
        num_epochs_no_improvement += 1
        if num_epochs_no_improvement >= patience:
            print(f'Early stopping after {epoch+1} epochs')
            break

# Save the model architecture and parameters
torch.save(model.state_dict(), model_pth)



