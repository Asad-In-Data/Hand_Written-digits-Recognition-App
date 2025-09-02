import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf 
import gradio as gr

data = tf.keras.datasets.mnist

(train_images,train_labels),(test_images,test_labels) = data.load_data()