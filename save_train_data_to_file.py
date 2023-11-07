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

# Load the data and transform to PyTorch tensors
# Very important parameter, defining the shift variable for left and right steering angle
delta = 0.2
features, labels = data_loading(delta,labels_file,features_directory)
# save the features and labels
np.save('./features.npy',features)
np.save('./labels.npy',labels)