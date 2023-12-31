
from dataloader import SteeringDataset
from networks import SteeringModel
import os

# Project root directory
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# data type, model type AND their classes
DATA_TYPE = 'combined_dataset_20231214'
MODEL_TYPE = 'ResNet50'
LOAD_TRAIN_MODEL_PATH = None

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

# load model path
LOAD_MODEL_PATH = ROOT_DIR+'/models/{}/{}/epho_7.pth'.format(DATA_TYPE,MODEL_TYPE)

#CUDA device
DEVICE = "cuda"

# Hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 0.0012
MAX_EPOCHS = 100
WEIGHT_DECAY = 0.0
PATIENCE = 5
DELTA = 0.2
BETAS = (0.9, 0.999)


