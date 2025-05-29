import numpy as np

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


def create_sequences(data, target_idx, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size, target_idx])
    return np.array(X), np.array(y)


def denormalize(scaler, y_pred_scaled, target_feature, columns):
    target_idx = list(columns).index(target_feature)
    dummy = np.zeros((len(y_pred_scaled), len(columns)))
    dummy[:, target_idx] = y_pred_scaled.flatten()
    return scaler.inverse_transform(dummy)[:, target_idx]