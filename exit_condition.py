#  exit_condition.py
#  Copyright (c) 2019 Aditya Rathod. All rights reserved.

from tensorflow import keras


class F1Callback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        """
        Calculate the F1 score after each epoch and exit after reaching some arbitrary score.
        :param epoch: The current epoch.
        :param logs: Logs of current metrics.
        :return:
        """
        p = logs['val_precision']
        r = logs['val_recall']
        f1 = 2 * ((p * r) / (p + r))
        print(f'\nval_f1: {f1}')
        if f1 >= 0.77:
            self.model.stop_training = True