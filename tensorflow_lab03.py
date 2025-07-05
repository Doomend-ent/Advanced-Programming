import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import time

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train / 255   # Fill in normalization factor
x_test = x_test / 255     # Fill in normalization factor
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28)),        # Fill input shape
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),  # Fill number of hidden neurons
    tf.keras.layers.Dense(10, activation='softmax')  # Fill number of output neurons
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',       # Fill name of loss function
              metrics=['accuracy'])

start = time.time()
model.fit(x_train, y_train, epochs=5)
end = time.time()
print(f"TF Training time: {end-start:.2f} seconds")       # Output training time
model.evaluate(x_test, y_test)

# Evaluate
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"TF Test Accuracy: {test_acc:.4f}")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("model.tflite", "wb") as f:
    f.write(tflite_model)