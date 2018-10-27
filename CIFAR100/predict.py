import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # Disable TF info logs.
from keras.models import load_model
from keras.preprocessing import image
import parameters
import keras
import numpy as np
import pickle
import sys
import pandas as pd
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt


if len(sys.argv) != 2:
    print("Usage: {} <image>".format(sys.argv[0]))
    exit(1)

def c100_classify(image, model):
    label_list_path = 'datasets/cifar-100-python/meta'
    keras_dir = os.path.expanduser(os.path.join('~', '.keras'))
    datadir_base = os.path.expanduser(keras_dir)
    if not os.access(datadir_base, os.W_OK):
        datadir_base = os.path.join('/tmp', '.keras')
    label_list_path = os.path.join(datadir_base, label_list_path)

    with open(label_list_path, mode='rb') as f:
        labels = pickle.load(f)
    # print(labels)
    prob = model.predict_proba(np.reshape(
        image, (1, 32, 32, 3)), batch_size=1, verbose=0)
    pred = pd.DataFrame(data=np.reshape(prob, 100), index=labels['fine_label_names'],
                        columns={'probability'}).sort_values('probability', ascending=False)
    pred['name'] = pred.index
    return pred[:5]


# Load the model and the test generator
try:
    model = load_model(parameters.SAVER_PATH)
except:
    print("The model couldn't be loaded.")
    exit(1)

image = image.load_img(sys.argv[1], target_size=(
    parameters.IMG_SIZE, parameters.IMG_SIZE))
image_small = misc.imresize(
    image, (parameters.IMG_SIZE, parameters.IMG_SIZE, 3)) / 255.
pred = c100_classify(image_small, model)
data = [{'name': x, 'probability': y}
        for x, y in zip(pred.iloc[:, 1], pred.iloc[:, 0])]

D = dict()
for i in data:
    D[i["name"]] = i["probability"]

for i in data:
    print("{} --> {}".format(i["name"],i["probability"]))

plt.bar(range(len(D)), list(D.values()), align='center')
plt.xticks(range(len(D)), list(D.keys()))

plt.show()
