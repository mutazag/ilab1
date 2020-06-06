#%%
from __future__ import absolute_import, division, print_function, unicode_literals
# Install TensorFlow
import tensorflow as tf
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

#%%
gpus = tf.config.experimental.list_physical_devices('GPU')
print(f'gpu device: {gpus}')
print(f'tf version: {tf.__version__}') 
#%%
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
history = model.fit(x_train, y_train, validation_split=0.33, epochs=20)
model.evaluate(x_test,  y_test, verbose=2)
#%%
def plot(history):
    import matplotlib.pyplot as plt
    print(history.history.keys())
    # "Loss"
    plt.subplot(1,2,1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['acc'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    plt.subplot(1,2,2)
    plt.plot(history.history['val_loss'])
    plt.plot(history.history['val_acc'])
    plt.title('model validation  loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

#%%
plot(history)

#%%
