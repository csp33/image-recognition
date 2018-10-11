import parameters
from keras.preprocessing.image import ImageDataGenerator


def get_dataset():
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    validation_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        parameters.TRAINING_FOLDER,
        target_size=(parameters.IMG_SIZE, parameters.IMG_SIZE),
        batch_size=parameters.BATCH_SIZE,
        class_mode='categorical')

    validation_generator = validation_datagen.flow_from_directory(
        parameters.VALIDATION_FOLDER,
        target_size=(parameters.IMG_SIZE, parameters.IMG_SIZE),
        batch_size=parameters.BATCH_SIZE,
        class_mode='categorical')
    return train_generator, validation_generator


def get_test_generator():
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    test_generator = validation_datagen.flow_from_directory(
        parameters.TEST_FOLDER,
        target_size=(parameters.IMG_SIZE, parameters.IMG_SIZE),
        batch_size=parameters.BATCH_SIZE,
        class_mode='categorical')
    return test_generator
