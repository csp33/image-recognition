import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # Disable TF info logs.
from keras.models import load_model
from keras.datasets import cifar100
import parameters
import keras
import time


# Load the model and the test generator
try:
    model = load_model(parameters.SAVER_PATH)
except:
    print("The model couldn't be loaded.")
    exit(1)

_, (x_test, y_test) = cifar100.load_data()
test_files=x_test.shape[0]
y_test = keras.utils.to_categorical(y_test, parameters.NUM_CLASSES)
x_test = x_test.astype('float32')
x_test /= 255.

# Evaluate it
score = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', score[0])
print('Test accuracy: ', score[1])

# Export the results

filename = "{}/{}.txt".format(parameters.STATS_PATH, time.strftime("%d_%m_%Y"))

with open(filename, 'w') as f:
    f.write('Number of test files: {}\n'.format(test_files))
    f.write('Test loss: {}\n'.format(score[0]))
    f.write('Test accuracy: {}\n'.format(score[1]))

print("Result succesfully exported to {}.".format(filename))
