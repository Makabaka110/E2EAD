import torch
import torch.nn as nn
import torch.optim as optim 
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import cv2
import matplotlib.pyplot as plt
from networks import SteeringModel
from dataloader import *


##########################################################################################
##########################################################################################
                        # 未经训练的数据的可视化 开始
##########################################################################################
##########################################################################################

#随机加载一张图片和标签进行可视化
# Visualize a random image and label from training set
# Define the file directory
features_directory = './data/'
labels_file = './data/driving_log.csv'

# # Load the data and transform to PyTorch tensors
# # Very important parameter, defining the shift variable for left and right steering angle
# delta = 0.2
# features, labels = data_loading(delta,labels_file,features_directory)

#load the features and labels from .npy
features = np.load('./features.npy')
labels = np.load('./labels.npy')

# #visualize 10 random features and labels in one picture
# fig = plt.figure(figsize=(30, 15))
# for i in range(10):
#     idx = np.random.randint(len(features))
#     ax = fig.add_subplot(2, 5, i+1)
#     ax.imshow(features[idx], cmap='gray')
#     ax.set_title('Steering Angle: {:.3f}'.format(labels[idx]))
#     ax.axis('off')
# plt.show()

##########################################################################################
##########################################################################################
                        # 未经训练的数据的可视化  结束
##########################################################################################
##########################################################################################




##########################################################################################
##########################################################################################
                        # 训练后的数据的可视化 开始
##########################################################################################
##########################################################################################

# load model.h5
model = SteeringModel()
# print(model)
model.load_state_dict(torch.load('./model.pth'))
model.eval()

# Split the data into train and validation sets
dataset = SteeringDataset(features, labels, transform=transforms.ToTensor())

# Define the data loaders
dataset_loader = DataLoader(dataset, batch_size=32, shuffle=True)

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

##########################################################################################
##########################################################################################
                        # 训练后的数据的可视化  结束
##########################################################################################
##########################################################################################

