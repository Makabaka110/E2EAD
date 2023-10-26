import csv
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Preprocess: change to HSV space and resize to rows = 16,cols = 32
def preprocess(img):
    resized = cv2.resize(img, (128, 64))
    return resized



# Load the left, center, and right camera data, shift (-/+) delta for left and right camera
def data_loading(delta,labels_file,features_directory):
    logs = []
    features = []
    labels = []

    with open(labels_file, 'rt') as f:
        reader = csv.reader(f)
        for line in reader:
            logs.append(line)
        log_labels = logs.pop(0)

    # Load the images and labels(only center camera)
    for i in range(len(logs)):
        img_path = logs[i][0]
        img_path = features_directory + img_path
        img = plt.imread(img_path)
        features.append(preprocess(img))
        labels.append(float(logs[i][3]))

    # Augment the data by vertically flipping the image
    features = np.concatenate((features, np.flipud(features)), axis=0)
    labels = np.concatenate((labels, -np.array(labels)), axis=0)

    return features, labels 