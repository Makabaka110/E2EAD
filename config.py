from dataloader import *
from networks import *

# data type, model type AND their classes
DATA_TYPE = 'Keyboard_10mins'
MODEL_TYPE = 'ResNet50'

DATA_TYPE_CLASS = SteeringDataset
MODEL_TYPE_CLASS = SteeringModel

#raw data path
DATA_PATH = './data/keyboard_10min/'
LABELS_FILE = './data/keyboard_10min/driving_log.csv'

# features and labels path
FEATURES = DATA_PATH + 'features.npy'
LABELS = DATA_PATH + 'labels.npy'

# model path
MODEL_PATH = './models/{}/{}/'.format(DATA_TYPE,MODEL_TYPE)

# load model path
LOAD_MODEL_PATH = './models/{}/{}/model.pth'.format(DATA_TYPE,MODEL_TYPE)

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


