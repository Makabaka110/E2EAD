import torch
import torchvision.transforms as transforms
from torch.utils.data import  DataLoader
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("..")
import config

model = config.MODEL_TYPE_CLASS()
model.load_state_dict(torch.load(config.LOAD_MODEL_PATH, map_location=torch.device('cuda')))
model.eval()

# Split the data into train and validation sets
dataset = config.DATA_TYPE_CLASS(np.load(config.FEATURES), np.load(config.LABELS), transform=transforms.ToTensor())

# Define the data loaders
dataset_loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)

# randomly visualize 10 predicted steering angles and ground truth steering angles in the dataloader
fig = plt.figure(figsize=(30, 15))
for i, data in enumerate(dataset_loader, 0):
    inputs, labels = data
    outputs = model(inputs)

    for j in range(10):
        idx = np.random.randint(len(inputs))
        ax = fig.add_subplot(2, 5, i*10+j+1)
        ax.imshow(inputs[idx].permute(1, 2, 0).squeeze().numpy())
        ax.set_title('Predicted: {:.3f}\nGround Truth: {:.3f}'.format(outputs[idx].item(), labels[idx].item()))
        ax.axis('off')
    if i == 0:
        break
plt.show()
