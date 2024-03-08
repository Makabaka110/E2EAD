import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.utils as vutils
from PIL import Image

# Define custom dataset
class SteeringDatasetLSTM(Dataset):
    def __init__(self, features, labels, transform=None, sequence_length=5):
        self.features = features
        self.labels = labels
        self.transform = transform
        self.sequence_length = sequence_length
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
        return len(self.features) - self.sequence_length + 1

    def __getitem__(self, idx):
        img_idx = idx//2
        transform_idx = idx%2
        #get the img with img_idx to idx+self.sequence_length
        for i in range(idx, idx+self.sequence_length):
            img = Image.open(self.features[i])
            if transform_idx == 0:
                img = self.transform_no_flip(img)
                label = self.labels[i]
            elif transform_idx == 1:
                img = self.transform_flip(img)
                label = -self.labels[i]
            # add img to sequence
            sequence_features.append(img)
        
        if transform_idx == 0:
            sequence_label = self.labels[idx+self.sequence_length-1]
        elif transform_idx == 1:
            sequence_label = -self.labels[idx+self.sequence_length-1]

        if self.transform:
            sequence_features = torch.stack([self.transform(feature) for feature in sequence_features])
        return sequence_features, sequence_label