import os
import csv
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Preprocess: change to HSV space and resize to rows = 224,cols = 224
def preprocess(img):
    resized = cv2.resize(img, (224, 224))
    return resized



# Load the left, center, and right camera data, shift (-/+) delta for left and right camera
def combined_data_loading(delta,data_directory):
    sub_directory = []
    features = []
    labels = []
    # find all visible sub directory under data_directory
    for name in os.listdir(data_directory):
        if not name.startswith('.') and os.path.isdir(os.path.join(data_directory, name)):
            sub_directory.append(name)
    # if there is one sub directory called IMG, then the sub_directory has only one element 
    # which is exactly the data_directory itself
    if len(sub_directory) == 1 and sub_directory[0] == 'IMG':
        sub_directory = ['']
    print('combined data contains:')
    print(sub_directory)


    for sub_dir in sub_directory:
        features_directory = data_directory + sub_dir + '/'
        labels_file = data_directory + sub_dir + '/driving_log.csv'
        logs = []

        print('extracting data from: ' + sub_dir + '    started')
    
        with open(labels_file, 'rt') as f:
            reader = csv.reader(f)
            for line in reader:
                logs.append(line)
            log_labels = logs.pop(0)

        # Load the images and labels(only center camera)
        for i in range(len(logs)):
            for j in range(3):
                img_path = logs[i][j]
                img_path = features_directory + img_path
                img = plt.imread(img_path)
                features.append(preprocess(img))
                if j == 0:
                    labels.append(float(logs[i][3]))
                elif j == 1:
                    labels.append(float(logs[i][3])+delta)
                elif j == 2:
                    labels.append(float(logs[i][3])-delta)

        print('extracting data from: ' + sub_dir + '    finished')

    print('doing data augmentation   started')

    # Augment the data by vertically flipping the image
    features = np.concatenate((features, np.flipud(features)), axis=0)
    labels = np.concatenate((labels, -np.array(labels)), axis=0)

    print('doing data augmentation   finished')


    return features, labels 