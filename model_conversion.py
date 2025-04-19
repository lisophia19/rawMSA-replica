import pandas as pd
import os
import tensorflow as tf
print(tf.__version__)

#run from scripts folder
with open('rawmsa-original/models/ss/list', 'r') as file:
    model_list = file.read().splitlines()

print(model_list[:2])

for model_path in model_list[:1]:
    model_path = "rawmsa-original/models/ss/" + model_path[10:]
    model = tf.keras.models.load_model(model_path)
    model.summary()

    #get more information abt layer parameters
    for layer in model.layers:
        print(layer.name, layer.__class__.__name__, layer.get_config())

