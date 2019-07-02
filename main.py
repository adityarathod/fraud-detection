#  main.py
#  Copyright (c) 2019 Aditya Rathod. All rights reserved.

from tensorflow import keras
from load_data import load_data
from exit_condition import AccuracyCallback

(x_train, y_train), (x_test, y_test), (x_val, y_val) = load_data()

print('Training set size:', x_train.shape[0])

model: keras.models.Model = keras.Sequential([
    keras.layers.Dense(5, activation='relu', input_shape=(29,)),
    keras.layers.Dense(1, activation='sigmoid')
])

cb = AccuracyCallback()

print(x_train.shape)

model.compile('adam', loss='binary_crossentropy', metrics=['acc', keras.metrics.Precision(), keras.metrics.Recall()])
model.fit(x_train, y_train, epochs=20, batch_size=64, validation_data=(x_val, y_val), callbacks=[cb])

print('Training complete.')

print('Evaluation on test set:')
test_metrics = model.evaluate(x_test, y_test)
test_f1 = 2 * ((test_metrics[2] * test_metrics[3]) / (test_metrics[2] + test_metrics[3]))
print(f'f1: {test_f1}')

model.save(f'models/ckpt-f1-{test_f1}-acc-{test_metrics[1]}-loss-{test_metrics[0]}.h5')