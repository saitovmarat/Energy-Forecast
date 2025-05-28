import numpy as np
import pandas as pd

def train_val_test_split(data, val_ratio, test_ratio):
    n = len(data)
    test_end = n
    test_start = n - int(n * test_ratio)
    val_end = test_start
    val_start = val_end - int(n * val_ratio)
    
    train = data[:val_start]
    val = data[val_start:val_end]
    test = data[test_start:test_end]
    
    return train, val, test


def create_one_step_sequences(data, target_idx, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size, target_idx])
    return np.array(X), np.array(y)

def create_multi_step_sequences(data, target_feature, window_size, n_steps_ahead):
    X, y = [], []
    for i in range(len(data) - window_size - n_steps_ahead + 1):
        X.append(data[i:(i + window_size)])
        y.append(data[i + window_size:i + window_size + n_steps_ahead][target_feature])
    return np.array(X), np.array(y)


def denormalize_one_step(scaler, y_pred_scaled, target_feature, columns):
    dummy = pd.DataFrame(np.zeros((len(y_pred_scaled), len(columns))), columns=columns)
    dummy[target_feature] = y_pred_scaled.flatten()
    return scaler.inverse_transform(dummy)[:, columns.get_loc(target_feature)]

def denormalize_multi_step(scaler, y_pred_scaled, target_feature, columns, n_steps):
    dummy = pd.DataFrame(np.zeros((len(y_pred_scaled), len(columns))), columns=columns)
    preds = []
    for i in range(n_steps):
        dummy[target_feature] = y_pred_scaled[:, i]
        denorm = scaler.inverse_transform(dummy)[:, columns.get_loc(target_feature)]
        preds.append(denorm)
    return np.array(preds).T 