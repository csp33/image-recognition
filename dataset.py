import cv2
import os
import glob
from sklearn.utils import shuffle
import numpy as np


def load_train(train_path, img_size, categories):
    # Create the arrays
    images = []
    labels = []
    img_names = []
    description = []

    # For each category
    for field in categories:
        index = categories.index(field)
        path = os.path.join(train_path, field, '*g')
        files = glob.glob(path)
        # For each img
        for file in files:
            image = cv2.imread(file)
            image = cv2.resize(
                image, (img_size, img_size), 0, 0, cv2.INTER_LINEAR)
            image = image.astype(np.float32)
            image = np.multiply(image, 1.0 / 255.0)
            images.append(image)
            label = np.zeros(len(categories))
            label[index] = 1.0
            labels.append(label)
            flbase = os.path.basename(file)
            img_names.append(flbase)
            description.append(field)
        images = np.array(images)
        labels = np.array(labels)
        img_names = np.array(img_names)
        description = np.array(description)

        return images, labels, img_names, description


class Dataset(object):
    # Constructor
    def __init__(self, imgs, labels, img_names, description):
        self._num_examples = imgs.shape[0]
        self._images = imgs
        self._labels = labels
        self._img_names = img_names
        self._description = description
        self._completed_epochs = 0
        self._index_in_epoch = 0

    # Get methods
    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def img_names(self):
        return self._img_names

    @property
    def description(self):
        return self._description

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_done(self):
        return self._completed_epochs

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size

        if self._index_in_epoch > self._num_examples:
            # After each epoch we update this
            self._completed_epochs += 1
            start = 0
            self._index_in_epoch = batch_size
            #assert batch_size <= self._num_examples
        end = self._index_in_epoch

        return self._images[start:end], self._labels[start:end], self._img_names[start:end], self._description[start:end]


def read_train_set(train_path, img_size, categories, val_size):
    class Datasets(object):
        pass
    data_sets = Datasets()

    images, labels, img_names, description = load_train(
        train_path, img_size, categories)
    images, labels, img_names, description = shuffle(
        images, labels, img_names, description)

    # Number of validation images.
    val_size = int(val_size * images.shape[0])

    # From 0 to val_size -> Validation images
    val_images = images[:val_size]
    val_labels = labels[:val_size]
    val_img_names = img_names[:val_size]
    val_description = description[:val_size]

    # From val_size to the end -> Training

    train_images = images[val_size:]
    train_labels = labels[val_size:]
    train_img_names = img_names[val_size:]
    train_description = description[val_size:]

    # Create the datasets
    data_sets.train = Dataset(
        train_images, train_labels, train_img_names, train_description)
    data_sets.valid = Dataset(
        val_images, val_labels, val_img_names, val_description)

    return data_sets
