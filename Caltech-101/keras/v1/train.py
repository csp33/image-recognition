from __future__ import print_function
from tensorflow import keras

print(keras.__version__)

from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.optimizers import SGD, RMSprop, Adam
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import parameters  # Configurable file
from numpy.random import seed


# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    parameters.TRAINING_FOLDER,
    target_size=(parameters.IMG_SIZE, parameters.IMG_SIZE),
    batch_size=parameters.BATCH_SIZE,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    parameters.VALIDATION_FOLDER,
    target_size=(parameters.IMG_SIZE, parameters.IMG_SIZE),
    batch_size=parameters.BATCH_SIZE,
    class_mode='categorical')

num_classes = len(train_generator.class_indices)

model = Sequential()
#model.add(Dense(320, activation='sigmoid', input_shape=(784,)))
#model.add(Dense(num_classes, activation='sigmoid'))

model.add(Conv2D(32, (3, 3), input_shape = (parameters.IMG_SIZE, parameters.IMG_SIZE, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Flatten())
model.add(Dense(units = 128, activation = 'relu'))
model.add(Dense(units = 1, activation = 'sigmoid'))


model.compile(loss='mean_squared_error',
              # loss='categorical_crossentropy',
              # optimizer=RMSprop(lr=0.001),
              optimizer=SGD(lr=parameters.LEARNING_RATE, momentum=0.0),
              # momentum=0.2),
              metrics=['accuracy'])

nb_train_samples = len(train_generator.filenames)
nb_validation_samples = len(validation_generator.filenames)


history = model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // parameters.BATCH_SIZE,
    epochs=parameters.EPOCHS,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // parameters.BATCH_SIZE, verbose=1)

# summarize history for accuracy

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
