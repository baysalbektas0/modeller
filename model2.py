# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 22:50:12 2021

@author: BektasBaysal
"""
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.python.keras import backend as K
from tensorflow.compat.v1 import get_default_graph
from typing import Union
import numpy as np

model = VGG16(weights='imagenet')

config =  tf.compat.v1.ConfigProto()
session = tf.compat.v1.Session(config=config)
K.set_session(session)

graph = get_default_graph()

def predict(image: np.ndarray) -> Union[str, None]:

    try:
        with graph.as_default():
            K.set_session(session)
            preds = model.predict(image)

        prediction = str(decode_predictions(preds, top=1)[0])

        return prediction
    except Exception as e:
        print(e)
        return None