from tensorflow import keras # Keras framework

######## Keras components ##########
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.optimizers import SGD, RMSprop, Adam
###################################
import matplotlib.pyplot as plt
import dataset # Function to get the training & validation data
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

model.add(Conv2D(32, (3, 3), input_shape=(parameters.IMG_SIZE,
                                          parameters.IMG_SIZE, 3),
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))

###############################

##### Model compulation #######

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

###############################

###### Model evaluation #######

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

###############################
