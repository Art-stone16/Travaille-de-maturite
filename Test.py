import numpy as np
import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import keras

(x_train,y_train),(x_test,y_test) = keras.datasets.mnist.load_data()
# Fait en sorte que les images soie entre 0 et 1
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
# Fait en sorte que les images soient en 28 x 28
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")

model = keras.saving.load_model("final_model.keras")
# Assume 'model' is your compiled and trained Keras model
# Assume 'x_test' and 'y_test' are your test features and labels

print("Evaluating model on test data...")
results = model.evaluate(x_test, y_test, batch_size=128, verbose=1)

# The 'results' variable contains the loss and metric values
# The order corresponds to model.metrics_names
print("Test Loss:", results[0])
print("Test Accuracy:", results[1]) # Assuming accuracy was the first metric specified

# Alternatively, if you compiled with named metrics:
# results_dict = model.evaluate(x_test, y_test, return_dict=True)
# print(results_dict)

