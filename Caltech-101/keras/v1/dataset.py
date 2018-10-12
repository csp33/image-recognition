import parameters
from keras.preprocessing.image import ImageDataGenerator


train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

validation_datagen = ImageDataGenerator(rescale=1. / 255)

test_datagen = ImageDataGenerator(rescale=1. / 255)


def get_dataset():
    try:
        train_generator = train_datagen.flow_from_directory(
            parameters.TRAINING_FOLDER,
            # 128,128,3 ??
            color_mode="rgb",
            target_size=(parameters.IMG_SIZE, parameters.IMG_SIZE),
            batch_size=parameters.BATCH_SIZE,
            class_mode='categorical',
            shuffle=True,
            seed=42)
        # Should be categorical !!!
    except:
        print("The training data couldn't be loaded.")
        exit(1)
    try:
        validation_generator = validation_datagen.flow_from_directory(
            parameters.VALIDATION_FOLDER,
            color_mode="rgb",
            target_size=(parameters.IMG_SIZE, parameters.IMG_SIZE),
            batch_size=parameters.BATCH_SIZE,
            class_mode='categorical',
            shuffle=True,
            seed=42)
        # Should be categorical !!!
    except:
        print("The validation data couldn't be loaded.")
        exit(1)
    return train_generator, validation_generator



def get_test_generator():
    try:
        test_generator = test_datagen.flow_from_directory(
            parameters.TEST_FOLDER,
            color_mode="rgb",
            target_size=(parameters.IMG_SIZE, parameters.IMG_SIZE),
            batch_size=parameters.BATCH_SIZE,
            class_mode='categorical')
        # Should be categorical !!!
    except:
        print("The validation data couldn't be loaded.")
        exit(1)
    return test_generator


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
