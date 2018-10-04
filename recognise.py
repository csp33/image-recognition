import tensorflow as tf
import numpy as np
import os
import glob
import cv2
import sys
import argparse
import parameters
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Disable TF debug info


if len(sys.argv) != 2:
    print("Use: {} <image>".format(sys.argv[0]))
    exit(-1)
# First, retrieve the path of the image

dir_path = os.path.dirname(os.path.realpath(__file__))
image_path = sys.argv[1]
filename = dir_path + '/' + image_path

images = []
# Read the image with OpenCV and resize it

image = cv2.imread(filename)
image = cv2.resize(image, (parameters.IMG_SIZE,
                           parameters.IMG_SIZE), 0, 0, cv2.INTER_LINEAR)

images.append(image)
images = np.array(images, dtype=np.uint8)
images = images.astype('float32')
images = np.multiply(images, 1.0 / 255.0)
# The input to the network is of shape [None image_size image_size num_channels]. Hence we reshape.
x_batch = images.reshape(1, parameters.IMG_SIZE,
                         parameters.IMG_SIZE, parameters.NUM_CHANNELS)


# First, let's restore the saved model

session = tf.Session()

# Then, recreate the network graph

saver = tf.train.import_meta_graph('./saver/image_recognition.meta')

# Now, restore the checkpoint

saver.restore(session, tf.train.latest_checkpoint('./saver'))

# Load the graph

graph = tf.get_default_graph()

# Now, let's get hold of the op that we can be processed to get the output.
# In the original network y_pred is the tensor that is the prediction of the network
y_pred = graph.get_tensor_by_name("y_pred:0")

# Let's feed the images to the input placeholders
x = graph.get_tensor_by_name("x:0")
y_true = graph.get_tensor_by_name("y_true:0")
y_test_images = np.zeros((1, len(os.listdir(parameters.TRAINING_FOLDER))))


# Creating the feed_dict that is required to be fed to calculate y_pred
feed_dict_testing = {x: x_batch, y_true: y_test_images}
result = session.run(y_pred, feed_dict=feed_dict_testing)


categories = os.listdir(parameters.TRAINING_FOLDER)

prediction=result[0][0]
prediction_idx=0


for i in range(1,len(result[0])):
    if prediction < result[0][i]:
        prediction=result[0][i]
        prediction_idx=i

print(categories[prediction_idx])
