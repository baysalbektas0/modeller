# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 17:03:14 2021

@author: BektasBaysal
"""

from typing import Union
import numpy as np
import tensorflow as tf
from tensorflow.compat.v1 import get_default_graph
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.python.keras import backend as K

config =  tf.compat.v1.ConfigProto()
session = tf.compat.v1.Session(config=config)
K.set_session(session)

graph = get_default_graph()

model = ResNet50(weights="imagenet")


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
