from tensorflow import keras

class AccuracyCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs['f1'] > 0.3:
            self.model.stop_training = True