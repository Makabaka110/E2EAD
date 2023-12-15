import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.utils as vutils
from PIL import Image

# Define custom dataset
class SteeringDataset(Dataset):
    def __init__(self, features, labels, transform=None):
        self.features = features
        self.labels = labels
        self.transform_no_flip = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize the image to 224x224 pixels
            transforms.ToTensor(),  # Convert the image to a PyTorch tensor
        ])
        self.transform_flip = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize the image to 224x224 pixels
            transforms.RandomHorizontalFlip(1.0),  # Always flip the image vertically
            transforms.ToTensor(),  # Convert the image to a PyTorch tensor
        ])

    def __len__(self):
        return 2*len(self.features)

    def __getitem__(self, idx):
        img_idx = idx//2
        transform_idx = idx%2
        img = Image.open(self.features[img_idx])  # Load the image as a PIL Image

        if transform_idx == 0:
            img = self.transform_no_flip(img)
            label = self.labels[img_idx]
        elif transform_idx == 1:
            img = self.transform_flip(img)
            label = -self.labels[img_idx]
        
        # if self.transform:
        #     feature = self.transform(feature)
            
        return img, label