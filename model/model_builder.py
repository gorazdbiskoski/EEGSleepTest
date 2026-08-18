import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.backend import clear_session

N_CLASSES = 5

def prepare_data(X):
    if X.ndim == 3 and X.shape[1] < X.shape[2]:
        X = np.transpose(X, (0, 2, 1))
    return X


def build_model(params, input_shape):
    filters, kernel_size, lstm_units, dropout, lr, batch_size = params

    filters = int(filters)
    kernel_size = int(kernel_size)
    lstm_units = int(lstm_units)
    batch_size = int(batch_size)

    clear_session()

    model = Sequential([
        Conv1D(filters=filters, kernel_size=kernel_size, activation="relu", padding="same", strides=2,
               input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(dropout),
        Conv1D(filters=filters, kernel_size=kernel_size, activation="relu", padding="same", strides=2),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(dropout),
        LSTM(lstm_units),
        Dense(32, activation="relu"),
        Dropout(dropout),
        Dense(N_CLASSES, activation="softmax")
    ])

    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def evaluate_model(
    params,
    X_train,
    X_val,
    y_train,
    y_val,
    class_weight=None
):
    batch_size = int(params[5])

    X_train = prepare_data(X_train)
    X_val = prepare_data(X_val)

    model = build_model(params, X_train.shape[1:])

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=2,
        restore_best_weights=True
    )

    history = model.fit(
        X_train,
        y_train,
        epochs=4,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        class_weight=class_weight,
        callbacks=[early_stopping],
        verbose=1
    )

    result = {
        "val_loss": min(history.history["val_loss"]),
        "val_accuracy": max(history.history["val_accuracy"]),
        "epochs": len(history.history["loss"])
    }

    clear_session()

    return result