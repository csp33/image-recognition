import parameters
from keras.preprocessing.image import ImageDataGenerator


train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

validation_datagen = ImageDataGenerator(rescale=1. / 255)

test_datagen = ImageDataGenerator(rescale=1. / 255)


def get_dataset():
    train_generator = train_datagen.flow_from_directory(
        parameters.TRAINING_FOLDER,
        # 128,128,3 ??
        target_size=(parameters.IMG_SIZE, parameters.IMG_SIZE),
        batch_size=parameters.BATCH_SIZE,
        class_mode='binary')
    # Should be categorical !!!

    validation_generator = validation_datagen.flow_from_directory(
        parameters.VALIDATION_FOLDER,
        target_size=(parameters.IMG_SIZE, parameters.IMG_SIZE),
        batch_size=parameters.BATCH_SIZE,
        class_mode='binary')
    # Should be categorical !!!
    return train_generator, validation_generator


"""
class_mode: One of "categorical", "binary", "sparse", "input", or None.
Default: "categorical". Determines the type of label arrays that are returned:
"categorical" will be 2D one-hot encoded labels,
"binary" will be 1D binary labels, "sparse" will be 1D integer labels,
"input" will be images identical to input images (mainly used to work with autoencoders).
If None, no labels are returned (the generator will only yield batches of image data,
which is useful to use with  model.predict_generator(),  model.evaluate_generator(), etc.).
Please note that in case of class_mode None, the data still needs to reside in a subdirectory
of directory for it to work correctly.
"""


def get_test_generator():
    test_generator = test_datagen.flow_from_directory(
        parameters.TEST_FOLDER,
        target_size=(parameters.IMG_SIZE, parameters.IMG_SIZE),
        batch_size=parameters.BATCH_SIZE,
        class_mode='binary')
    # Should be categorical !!!
    return test_generator
