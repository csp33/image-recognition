import tensorflow as tf
import time
import math
import random
import os
import numpy
import dataset  # To manage the datasets
import parameters  # Configurable file
import cnn  # CNN creation utility

from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)


categories = os.listdir(parameters.TRAINING_FOLDER)
num_categories = len(categories)
val_size = 0.15  # 15% of the training set will be used as validation.
batch_size = 16
data = dataset.read_train_set(
    parameters.TRAINING_FOLDER, parameters.IMG_SIZE, categories, val_size)

print("Input data succesfully read.")
print("Number of training files: {}".format(len(data.train.labels)))
print("Number of validation files: {}".format(len(data.train.labels)))


# Create the placeholders


# For the training images
x = tf.placeholder(tf.float32, shape=[
    None, parameters.IMG_SIZE, parameters.IMG_SIZE, parameters.NUM_CHANNELS], name='x')

# For the predicions & the correct descriptions
y_true = tf.placeholder(
    tf.float32, shape=[None, num_categories], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)


# Now, let's create the net:

last_layer = cnn.create_neural_net(x, num_categories)

# Predicted probability of each class for each input image

y_pred = tf.nn.softmax(last_layer, name='y_pred')

# The class having higher probability is the prediction of the network.

y_pred_cls = tf.argmax(y_pred, dimension=1)

# Let's start the session

session = tf.Session()

session.run(tf.global_variables_initializer())

# Calculate the cost

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=last_layer,
                                                        labels=y_true)
cost = tf.reduce_mean(cross_entropy)

# OPTIMIZATION -> TensorFlow feature

optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

# Calculate the accuracy

correct_prediction = tf.equal(y_pred_cls, y_true_cls)

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


session.run(tf.global_variables_initializer())

# Function to show the progress of the training


def show_progress(epoch, feed_dict_train, feed_dict_validate, val_loss):
    acc = session.run(accuracy, feed_dict=feed_dict_train)
    val_acc = session.run(accuracy, feed_dict=feed_dict_validate)
    print(parameters.MSG.format(epoch + 1, acc, val_acc, val_loss))



saver = tf.train.Saver()

# Let's train the net:


def train(num_iteration):
    for i in range(0, num_iteration):
        x_batch, y_true_batch, _, cls_batch = data.train.next_batch(batch_size)
        x_valid_batch, y_valid_batch, _, valid_cls_batch = data.valid.next_batch(
            batch_size)

        feed_dict_tr = {x: x_batch,
                        y_true: y_true_batch}
        feed_dict_val = {x: x_valid_batch,
                         y_true: y_valid_batch}

        session.run(optimizer, feed_dict=feed_dict_tr)

        if i % int(data.train.num_examples / batch_size) == 0:
            val_loss = session.run(cost, feed_dict=feed_dict_val)
            epoch = int(i / int(data.train.num_examples / batch_size))
            show_progress(epoch, feed_dict_tr, feed_dict_val, val_loss)
            saver.save(session, './saver/image_recognition')


# Finally, execute the script

train(num_iteration=50000)


# First step: data gathering and cleaning


# Second step: preprocessing, feature extracion (if needed)


# Third step: model construction, iterations of parameter estimations (training)
