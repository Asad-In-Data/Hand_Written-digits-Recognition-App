import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf 
import gradio as gr

data = tf.keras.datasets.mnist

(train_images,train_labels),(test_images,test_labels) = data.load_data()
# Normalizing
train_images=train_images.reshape((60000,28,28,1)).astype("float32")/255 # reshaping because the model expects 4D input 
test_images=test_images.reshape((10000,28,28,1)).astype("float32")/255 # used float32 to reduce memory usage

train_labels= tf.keras.utils.to_categorical(train_labels) # hot encoding because the labels are categorical
test_labels= tf.keras.utils.to_categorical(test_labels) # one-hot encoding

# Building the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)), # 32 filters, 3x3 kernel size
    tf.keras.layers.MaxPooling2D((2, 2)), # 2x2 pooling which is used to downsample the feature maps
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'), # 64 filters, 3x3 kernel size # filters are used for feature extraction
    tf.keras.layers.MaxPooling2D((2, 2)), # 2x2 pooling which is used to downsample the feature maps
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(), 
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5, batch_size=64, validation_split=0.2)
model.evaluate(test_images, test_labels)
model.save('model.keras')
