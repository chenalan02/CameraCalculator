import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import itertools
from confusionmatrix import plot_confusion_matrix
from sklearn.metrics import confusion_matrix

from tensorflow.keras.layers import Dense, Input, Conv2D, BatchNormalization, MaxPool2D, Flatten, GlobalMaxPool2D
from tensorflow.keras.models import Model

CATEGORIES = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 'add': 10, 'dec': 11, 'div': 12, 'mul': 13, 'sub': 14}

data_generator = ImageDataGenerator(rescale = 1/255, validation_split = 0.33)

train_data = data_generator.flow_from_directory('dataset', target_size=(100,100),
    classes = CATEGORIES, subset = 'training', color_mode = 'grayscale')

val_data = data_generator.flow_from_directory('dataset', target_size=(100,100), 
    batch_size = 1, classes = CATEGORIES, subset = 'validation', color_mode = 'grayscale')


i = tf.keras.layers.Input(shape = (100, 100, 1))

x = tf.keras.layers.Conv2D(32, (2,2), padding='same', activation='relu')(i)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Conv2D(32, (2,2), padding='same', activation='relu')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.MaxPooling2D((2,2))(x)

x = tf.keras.layers.Conv2D(64, (2,2), padding='same', activation='relu')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Conv2D(64, (2,2), padding='same', activation='relu')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.MaxPooling2D((2,2))(x)

x = tf.keras.layers.Conv2D(128, (2,2), padding='same', activation='relu')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Conv2D(128, (2,2), padding='same', activation='relu')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.MaxPooling2D((2,2))(x)

x = tf.keras.layers.GlobalMaxPool2D()(x)
x = tf.keras.layers.Dropout(0.1)(x)
x = tf.keras.layers.Dense(258, activation='relu')(x)
x = tf.keras.layers.Dense(len(CATEGORIES), activation='softmax')(x)

model = tf.keras.models.Model(i, x)

model.compile(optimizer = 'adam',
              loss = 'categorical_crossentropy',
              metrics = ['accuracy'])

callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy', patience=10, restore_best_weights=True
)

r = model.fit(train_data, validation_data=val_data, epochs=40, callbacks=[callback])

plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

plt.plot(r.history['accuracy'], label='acc')
plt.plot(r.history['val_accuracy'], label='val_acc')
plt.legend()
plt.show()

model.save('MathSymbolRecognizerModel_1.h5')

val_data = data_generator.flow_from_directory('dataset', target_size=(100,100), 
    batch_size = 1, classes = CATEGORIES, subset = 'validation', color_mode = 'grayscale', shuffle=False)

Y_pred = model.predict(val_data)
y_pred = np.argmax(Y_pred, axis=1)
cm = confusion_matrix(val_data.classes, y_pred)
plot_confusion_matrix(cm, ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'add', 'dec', 'div', 'mul', 'sub']), 