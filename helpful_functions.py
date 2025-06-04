import numpy as np
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error

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


def train_and_evaluate_model(model, model_type, X_train, y_train, X_val, y_val, X_test, y_test, lr, window_size, scaler, target_feature, df_columns):
    model.compile(optimizer=Adam(learning_rate=lr), loss='mse')
    history = model.fit(X_train, y_train,
                        epochs=5,
                        batch_size=256,
                        validation_data=(X_val, y_val),
                        verbose=0)
    
    y_pred_val = model.predict(X_val, verbose=0)
    y_pred_test = model.predict(X_test, verbose=0)

    y_pred_val_dn = denormalize(scaler, y_pred_val, target_feature, df_columns)
    y_pred_test_dn = denormalize(scaler, y_pred_test, target_feature, df_columns)
    y_true_val_dn = denormalize(scaler, y_val.reshape(-1, 1), target_feature, df_columns)
    y_true_test_dn = denormalize(scaler, y_test.reshape(-1, 1), target_feature, df_columns)

    mse_val = mean_squared_error(y_true_val_dn, y_pred_val_dn)
    mse_test = mean_squared_error(y_true_test_dn, y_pred_test_dn)

    print(f"{model_type} | val_loss: {history.history['val_loss'][-1]:.4f} | MSE: {mse_val:.4f}")

    return {
        'model_type': model_type,
        'learning_rate': lr,
        'window_size': window_size,
        'train_loss': history.history['loss'][-1],
        'val_loss': history.history['val_loss'][-1],
        'mse_val_dn': mse_val,
        'mse_test_dn': mse_test
    }
