import numpy as np
import sys
sys.path.append("..")
# Import the custom classes
from dataloader import data_loading, combined_data_loading
from networks import *
import config

# load the features and labels 
features, labels = combined_data_loading(config.DELTA, config.DATA_PATH)

# save the features and labels as .npy files
print("saving the features and labels as .npy files  started")
np.save(config.FEATURES,features)
np.save(config.LABELS,labels)
print("saving the features and labels as .npy files  finished")