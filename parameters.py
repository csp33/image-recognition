
TRAINING_FOLDER = "./train_images"
IMG_SIZE = 128  # To be resized

# Define network graph params

FILTER_SIZE_CONV1 = 3
NUM_FILTERS_CONV1 = 32

FILTER_SIZE_CONV2 = 3
NUM_FILTERS_CONV2 = 32

FILTER_SIZE_CONV3 = 3
NUM_FILTERS_CONV3 = 64

FC_LAYER_SIZE = 128

NUM_CHANNELS = 3

MSG = "Training Epoch {0} --- Training Accuracy: {1:>6.1%}, Validation Accuracy: {2:>6.1%},  Validation Loss: {3:.3f}"
