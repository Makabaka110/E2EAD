import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
import csv
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
import os
from torch.utils.tensorboard import SummaryWriter

# Import the custom classes
from dataloader import *
from networks import *
import config 

writer = SummaryWriter()
# Split the data into train and validation sets
dataset = config.DATA_TYPE_CLASS(np.load(config.FEATURES), np.load(config.LABELS), transform=transforms.ToTensor())
train_dataset, val_dataset = random_split(dataset, [int(len(dataset)*0.9), len(dataset)-int(len(dataset)*0.9)])

# Define the data loaders
train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

model = config.MODEL_TYPE_CLASS()
if config.LOAD_TRAIN_MODEL_PATH:
    model.load_state_dict(torch.load(config.LOAD_TRAIN_MODEL_PATH))
# Use CUDA if available
device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)

# Optimize
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, betas=config.BETAS, eps=1e-08, weight_decay=config.WEIGHT_DECAY)

train_losses = []
val_losses = []

best_val_loss = np.inf
patience = config.PATIENCE
num_epochs_no_improvement = 0

print('Training started...')
for epoch in range(config.MAX_EPOCHS):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
     
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

    writer.add_scalars('Loss', {'Train': train_losses[-1],'Val':val_losses[-1]}, epoch)
    # writer.add_scalar('Loss/val', val_losses[-1], epoch)
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
    
    if not os.path.exists(config.MODEL_PATH):
        os.makedirs(config.MODEL_PATH)
    torch.save(model.state_dict(), config.MODEL_PATH+'epoch_'+str(epoch+1)+'.pth')

    #plt and save the loss curve image
    plt.plot(train_losses, label='Training loss')
    plt.plot(val_losses, label='Validation loss')
    plt.legend(frameon=False)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(config.MODEL_PATH+'loss.png')
    plt.close()

    
writer.close()

