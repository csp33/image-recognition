from keras.models import load_model
import parameters
import dataset
import time


# Load the model and the test generator

model = load_model(parameters.SAVER_PATH)
print("Model succesfully loaded.")
test_generator = dataset.get_test_generator()
print("Test data succesfully loaded.")
steps = len(test_generator.filenames) // parameters.BATCH_SIZE

# Evaluate it

score = model.evaluate_generator(test_generator, verbose=0, steps=steps)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Export the results

filename = "{}/{}.txt".format(parameters.STATS_PATH,time.strftime("%d_%m_%Y"))

with open(filename, 'w') as f:
    f.write('Number of test files: {}\n'.format(len(test_generator.filenames)))
    f.write('Test loss: {}\n'.format(score[0]))
    f.write('Test accuracy: {}\n'.format(score[1]))

print("Result succesfully exported.")
