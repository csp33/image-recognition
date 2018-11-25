from __future__ import print_function
import keras
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # Disable TF info logs.
import parameters
import time
from keras.datasets import cifar100
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import matplotlib.pyplot as plt

start = time.time()

# First, we must load the dataset

(x_train, y_train), (x_test, y_test) = cifar100.load_data()
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Convert to categorical
y_train = keras.utils.to_categorical(y_train, parameters.NUM_CLASSES)
y_test = keras.utils.to_categorical(y_test, parameters.NUM_CLASSES)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255.
x_test /= 255.

# Create the Convolutional net

model = Sequential()

model.add(Conv2D(128, (3, 3), padding='same',
                 input_shape=x_train.shape[1:], activation='elu'))
model.add(Conv2D(128, (3, 3), activation='elu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(256, (3, 3), padding='same', activation='elu'))
model.add(Conv2D(256, (3, 3), activation='elu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(512, (3, 3), padding='same', activation='elu'))
model.add(Conv2D(512, (3, 3), activation='elu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))


model.add(Flatten())
model.add(Dense(1024, activation='elu'))
model.add(Dropout(0.5))
model.add(Dense(parameters.NUM_CLASSES, activation='softmax'))

# Compile the model & show the summary

model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.rmsprop(lr=parameters.LEARNING_RATE,
                                                 decay=1e-6),
              metrics=['accuracy'])
model.summary()

# Preprocessing parameters:
datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    # randomly rotate images in the range (degrees, 0 to 180)
    rotation_range=0,
    # randomly shift images horizontally (fraction of total width)
    width_shift_range=0.1,
    # randomly shift images vertically (fraction of total height)
    height_shift_range=0.1,
    horizontal_flip=True,  # randomly flip images
    # vertical_flip=False)  # randomly flip images
    vertical_flip=True)  # randomly flip images

datagen.fit(x_train)

num_train_samples = x_train.shape[0]
num_validation_samples = x_test.shape[0]
train_steps = num_train_samples // parameters.BATCH_SIZE
validation_steps = num_validation_samples // parameters.BATCH_SIZE

history = model.fit_generator(datagen.flow(x_train, y_train,
                                           batch_size=parameters.BATCH_SIZE),
                              steps_per_epoch=train_steps,
                              epochs=parameters.EPOCHS,
                              validation_data=(x_test, y_test),
                              validation_steps=validation_steps)

current_time = time.strftime("%d_%m_%Y %H:%M:%S")
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

end = time.time()

print("Elapsed time: {} seconds.".format(end - start))

######## Model saving #########

model.save(parameters.SAVER_PATH)

###############################
