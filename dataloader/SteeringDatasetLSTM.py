import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.utils as vutils

# Define custom dataset
class SteeringDatasetLSTM(Dataset):
    def __init__(self, features, labels, transform=None, sequence_length=5):
        self.features = features
        self.labels = labels
        self.transform = transform
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.features) - self.sequence_length + 1

    def __getitem__(self, idx):
        sequence_features = self.features[idx:idx+self.sequence_length]
        sequence_label = self.labels[idx+self.sequence_length-1]  # label is the last one in the sequence
        if self.transform:
            sequence_features = torch.stack([self.transform(feature) for feature in sequence_features])
        return sequence_features, sequence_label