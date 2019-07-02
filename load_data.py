#  load_data.py
#  Copyright (c) 2019 Aditya Rathod. All rights reserved.

import pandas as pd


def load_data(path='data/train.csv'):
    X = pd.read_csv(path).sample(frac=1, random_state=1).reset_index(drop=True)
    y = X['Class']
    X.drop(columns=['Class', 'Time'], inplace=True)
    X = X.values
    y = y.values
    train_idx = int(round(X.shape[0] * 0.7, 0))
    test_idx = int(train_idx + round(X.shape[0] * 0.2, 0))
    val_idx = int(test_idx + round(X.shape[0] * 0.1, 0))
    x_train, y_train = X[:train_idx, :], y[:train_idx]
    x_test, y_test = X[train_idx:test_idx, :], y[train_idx:test_idx]
    x_val, y_val = X[test_idx:val_idx, :], y[test_idx:val_idx]
    return (x_train, y_train), (x_test, y_test), (x_val, y_val)