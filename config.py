
from dataloader import SteeringDataset
from networks import SteeringModel
import os

# Project root directory
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# data type, model type AND their classes

# Data type including 'keyboard_10min', 'github_data', 'xbox_5min', 'xbox_20min','github_xbox20','xbox_inverse_10min'
DATA_TYPE = 'github_xbox20' 
MODEL_TYPE = 'ResNet50'

DATA_TYPE_CLASS = SteeringDataset
MODEL_TYPE_CLASS = SteeringModel

#raw data path
DATA_PATH = ROOT_DIR + '/data/{}/'.format(DATA_TYPE)
LABELS_FILE = ROOT_DIR+ '/data/{}/delete_all_zero_driving_log.csv'.format(DATA_TYPE)

# features and labels path
FEATURES = DATA_PATH + 'features.npy'
LABELS = DATA_PATH + 'labels.npy'

# model path
MODEL_PATH = ROOT_DIR+'/models/{}/{}/'.format(DATA_TYPE,MODEL_TYPE)

# load drive model path
LOAD_MODEL_PATH = ROOT_DIR+'/models/{}/{}/epho_5.pth'.format(DATA_TYPE,MODEL_TYPE)

# load train model path
# LOAD_TRAIN_MODEL_PATH = ROOT_DIR+'/models/pretrained_model/epho_15.pth'
LOAD_TRAIN_MODEL_PATH = ''

#CUDA device
DEVICE = "cuda"

# Hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 0.0012
MAX_EPOCHS = 100
WEIGHT_DECAY = 0.0
PATIENCE = 5
DELTA = 0.25
BETAS = (0.9, 0.999)


