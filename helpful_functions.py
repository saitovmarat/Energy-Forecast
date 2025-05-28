import numpy as np
import pandas as pd

def create_multi_step_sequence(data, target_feature, window_size, n_steps_ahead):
    X, y = [], []
    for i in range(len(data) - window_size - n_steps_ahead + 1):
        X.append(data[i:(i + window_size)])
        y.append(data[i + window_size:i + window_size + n_steps_ahead][target_feature])
    return np.array(X), np.array(y)

def denormalize_multi_step(scaler, y_pred_scaled, target_feature, columns, n_steps):
    dummy = pd.DataFrame(np.zeros((len(y_pred_scaled), len(columns))), columns=columns)
    preds = []
    for i in range(n_steps):
        dummy[target_feature] = y_pred_scaled[:, i]
        denorm = scaler.inverse_transform(dummy)[:, columns.get_loc(target_feature)]
        preds.append(denorm)
    return np.array(preds).T 