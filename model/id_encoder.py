import tensorflow as tf
import numpy as np
from tensorflow.keras import Model

from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import os
import time
# from modules.evaluations import get_val_data, perform_val
from modules.models import ArcFaceModel
from modules.utils import set_memory_growth, load_yaml, l2_norm


class IDEncoder(Model):

    def __init__(self, args, model_path, intermediate_layers_names=None):
        super().__init__()
        self.args = args
        self.model = ArcFaceModel(size=112,backbone_type='ResNet50',training=False)
        ckpt_path = tf.train.latest_checkpoint(model_path)
        if ckpt_path is not None:
            self.model.load_weights(ckpt_path)
        else:
            print("[*] Cannot find ckpt.")
            exit()



    @tf.function
    def call(self, input_x, get_intermediate=False):
        embedding = self.model(input_x)
        embedding = tf.math.l2_normalize(embedding, axis=-1)
        embedding = tf.expand_dims(embedding, 1)

        return embedding
    
