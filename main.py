#  main.py
#  Copyright (c) 2019 Aditya Rathod. All rights reserved.

from tensorflow import keras

from exit_condition import F1Callback
from load_data import load_data

(x_train, y_train), (x_test, y_test), (x_val, y_val) = load_data()

print('Training set size:', x_train.shape[0])

# Define model
model: keras.models.Model = keras.Sequential([
    keras.layers.Dense(5, activation='relu', input_shape=(29,)),
    keras.layers.Dense(1, activation='sigmoid')
])

# Create post-epoch callback that will determine if we exit or not
cb = F1Callback()

# Compile and train model
model.compile('adam', loss='binary_crossentropy', metrics=['acc', keras.metrics.Precision(), keras.metrics.Recall()])
model.fit(x_train, y_train, epochs=20, batch_size=64, validation_data=(x_val, y_val), callbacks=[cb])

print('Training complete.')

# Evaluate on test set
print('Evaluation on test set:')
test_metrics = model.evaluate(x_test, y_test)
test_f1 = 2 * ((test_metrics[2] * test_metrics[3]) / (test_metrics[2] + test_metrics[3]))
print(f'f1: {test_f1}')

# Save model in models/ folder
model.save(f'models/ckpt-f1-{test_f1}-acc-{test_metrics[1]}-loss-{test_metrics[0]}.h5')