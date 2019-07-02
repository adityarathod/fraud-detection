#  exit_condition.py
#  Copyright (c) 2019 Aditya Rathod. All rights reserved.

from tensorflow import keras
import numpy as np
from sklearn.metrics import f1_score

class AccuracyCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        p = logs['val_precision']
        r = logs['val_recall']
        f1 = 2 * ((p * r) / (p + r))
        print(f'\nval_f1: {f1}')
        if f1 >= 0.77:
            self.model.stop_training = True