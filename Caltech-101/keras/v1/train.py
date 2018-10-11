from tensorflow import keras  # Keras framework

######## Keras components ##########
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.optimizers import SGD, RMSprop, Adam
###################################
import matplotlib.pyplot as plt
import dataset  # Function to get the training & validation data
import parameters  # Configurable file
from numpy.random import seed


# Print the Keras framework version used

print(keras.__version__)

# Load the dataset and set the variables

train_generator, validation_generator = dataset.get_dataset()

num_train_samples = len(train_generator.filenames)
num_validation_samples = len(validation_generator.filenames)

####### Model creation ########

model = Sequential()

# Input layer

model.add(Conv2D(32, (3, 3), input_shape=(parameters.IMG_SIZE,
                                          parameters.IMG_SIZE, 3),
                 activation='relu'))

# Pooling layer to reduce the size
model.add(MaxPooling2D(pool_size=(2, 2)))
# Flatten layer to convert to 1D vector
model.add(Flatten())
# Fully connected layer
model.add(Dense(units=128, activation='relu'))
# Output layer
model.add(Dense(units=1, activation='sigmoid'))

###############################

##### Model compilation #######

model.compile(loss='mean_squared_error',
              optimizer=SGD(lr=parameters.LEARNING_RATE, momentum=0.0),
              metrics=['accuracy'])

###############################

####### Model training ########

history = model.fit_generator(
    train_generator,
    steps_per_epoch=num_train_samples // parameters.BATCH_SIZE,
    epochs=parameters.EPOCHS,
    validation_data=validation_generator,
    validation_steps=num_validation_samples // parameters.BATCH_SIZE, verbose=1)

###############################

# Free some memory

del train_generator
del validation_generator

############ Plots ############

# Accuracy

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Loss

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


######## Model saving #########

model.save('./saver/image_recognition.h5')

###############################

###### Model evaluation #######

test_generator = parameters.get_test_generator()

score = model.evaluate_generator(
    test_generator, use_multiprocessing=True, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

###############################
