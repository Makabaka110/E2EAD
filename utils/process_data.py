import numpy as np
import sys
sys.path.append("..")
# Import the custom classes
from dataloader import data_loading
from networks import *
import config

# load the features and labels 
features, labels = data_loading(config.DELTA, config.LABELS_FILE, config.DATA_PATH)

# save the features and labels as .npy files
np.save(config.FEATURES,features)
np.save(config.LABELS,labels)