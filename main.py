from tensorflow import keras
from load_data import load_data
from exit_condition import AccuracyCallback
from custom_metrics import f1

(x_train, y_train), (x_test, y_test), (x_val, y_val) = load_data()

print('Training set size:', x_train.shape[0])

model = keras.Sequential([
    keras.layers.Dense(256, activation='relu', input_shape=(29,)),
    keras.layers.Dense(1, activation='sigmoid')
])

cb = AccuracyCallback()

model.compile('adam', loss='binary_crossentropy', metrics=['acc', f1])
model.fit(x_train, y_train, epochs=5, batch_size=64, validation_data=(x_val, y_val), callbacks=[cb])
print('Training complete.')
model.evaluate(x_test, y_test)