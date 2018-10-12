import keras  # Keras framework
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # Disable TF info logs.

######## Keras components ##########
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.optimizers import SGD, RMSprop, Adam
###################################
import matplotlib.pyplot as plt
import dataset  # Function to get the training & validation data
import parameters  # Configurable file
from numpy.random import seed
import time  # To print the date in the output file

# Print the Keras framework version used

print("Keras version: {}".format(keras.__version__))

# Load the dataset and set the variables

train_generator, validation_generator = dataset.get_dataset()

num_train_samples = len(train_generator.filenames)
num_validation_samples = len(validation_generator.filenames)
train_steps = num_train_samples // parameters.BATCH_SIZE
validation_steps = num_validation_samples // parameters.BATCH_SIZE

####### Model creation ########

model = Sequential()

# Input layer

model.add(Conv2D(32, (3, 3),
                 input_shape=(parameters.IMG_SIZE, parameters.IMG_SIZE, 3),
                 activation='relu'))

# Pooling layer to reduce the size

model.add(MaxPooling2D(pool_size=(2, 2)))
# Another convolutional layer

model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# Flatten layer to convert to 1D vector

model.add(Flatten())
# Fully connected layer

model.add(Dense(units=128, activation='relu'))
# Output layer

model.add(Dense(units=1, activation='sigmoid'))

"""
model = Sequential()
model.add(Dense(320, activation='sigmoid', input_shape=(784,)))
model.add(Dense(num_classes, activation='sigmoid'))

"""
###############################


##### Model compilation #######
model.compile(loss='mean_squared_error',
              optimizer=SGD(lr=parameters.LEARNING_RATE, momentum=0.0),
              metrics=['accuracy'])

model.summary()

###############################

####### Model training ########

history = model.fit_generator(
    train_generator, steps_per_epoch=train_steps, epochs=parameters.EPOCHS,
    validation_data=validation_generator, validation_steps=validation_steps,
    verbose=1)

###############################


############ Plots ############

# Accuracy

current_time = time.strftime("%d/%m/%Y %H:%M:%S")
current_date = time.strftime("%d_%m_%Y")

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model Accuracy ({})'.format(current_time))
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('{}/accuracy_{}.png'.format(parameters.STATS_PATH, current_date))
plt.close()
# Loss

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss ({})'.format(current_time))
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('{}/loss_{}.png'.format(parameters.STATS_PATH, current_date))


######## Model saving #########

model.save(parameters.SAVER_PATH)

###############################
