import tensorflow as tf
import parameters


# Initialize weights using a normal distribution with a small SD.


def create_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))


def create_biases(size):
    return tf.Variable(tf.constant(0.05, shape=[size]))


# CREATE NETWORK LAYERS

def create_convolutional_layer(input, num_input_channels, conv_filter_size, num_filters):
    # First, create the weights
    weights = create_weights(
        shape=[conv_filter_size, conv_filter_size, num_input_channels, num_filters])
    # Then, create the biases (also trained)
    biases = create_biases(num_filters)

    # Now, let's create the Convolutional Layer

    layer = tf.nn.conv2d(input=input, filter=weights,
                         strides=[1, 1, 1, 1], padding='SAME')
    layer += biases

    # Use the max pooling technique

    layer = tf.nn.max_pool(value=layer, ksize=[1, 2, 2, 1], strides=[
                           1, 2, 2, 1], padding='SAME')

    # The output of the pooling is fed to RELU, our activation function

    layer = tf.nn.relu(layer)

    return layer


def create_flatten_layer(layer):
    # First, retrieve the shape from the previous layer
    layer_shape = layer.get_shape()

    # number of features = height * width * NUM_CHANNELS
    num_features = layer_shape[1:4].num_elements()

    # Flatten the layer

    layer = tf.reshape(layer, [-1, num_features])

    return layer


def create_fully_connected_layer(input, num_inputs, num_outputs, use_relu=True):
    # First, create the weights and biases
    weights = create_weights(shape=[num_inputs, num_outputs])
    biases = create_biases(num_outputs)

    # Fully connected layer: x -> wx+b
    layer = tf.matmul(input, weights) + biases
    if use_relu:
        layer = tf.nn.relu(layer)
    return layer





def create_neural_net(x,num_categories):
    layer_conv1 = create_convolutional_layer(input=x,
                                             num_input_channels=parameters.NUM_CHANNELS,
                                             conv_filter_size=parameters.FILTER_SIZE_CONV1,
                                             num_filters=parameters.NUM_FILTERS_CONV1)
    layer_conv2 = create_convolutional_layer(input=layer_conv1,
                                             num_input_channels=parameters.NUM_FILTERS_CONV1,
                                             conv_filter_size=parameters.FILTER_SIZE_CONV2,
                                             num_filters=parameters.NUM_FILTERS_CONV2)

    layer_conv3 = create_convolutional_layer(input=layer_conv2,
                                             num_input_channels=parameters.NUM_FILTERS_CONV2,
                                             conv_filter_size=parameters.FILTER_SIZE_CONV3,
                                             num_filters=parameters.NUM_FILTERS_CONV3)

    layer_flat = create_flatten_layer(layer_conv3)

    layer_fc1 = create_fully_connected_layer(input=layer_flat,
                                            num_inputs=layer_flat.get_shape()[
                                                1:4].num_elements(),
                                            num_outputs=parameters.FC_LAYER_SIZE,
                                            use_relu=True)

    layer_fc2 = create_fully_connected_layer(input=layer_fc1,
                                            num_inputs=parameters.FC_LAYER_SIZE,
                                            num_outputs=num_categories,
                                            use_relu=False)
    return layer_fc2
