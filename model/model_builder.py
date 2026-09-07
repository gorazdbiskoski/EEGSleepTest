import numpy as np

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
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

    # EarlyStopping(restore_best_weights=True) puts the best-val_loss epoch's
    # weights back before fit() returns -- and does so whether or not it
    # actually stopped early -- so this single predict describes that one
    # epoch. Reading max(val_accuracy) out of history instead could report a
    # different epoch than the min(val_loss) being optimised against.
    y_pred = model.predict(X_val, batch_size=batch_size, verbose=0).argmax(axis=1)

    # Macro averaging weighs every sleep stage equally, so N1 -- the rarest and
    # hardest stage -- is not drowned out by W and N2. zero_division=0 keeps a
    # degenerate model that never predicts some stage from writing NaN into the
    # results CSVs and convergence plots.
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_val, y_pred, average="macro", zero_division=0
    )

    result = {
        "val_loss": min(history.history["val_loss"]),
        "val_accuracy": float(accuracy_score(y_val, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "epochs": len(history.history["loss"])
    }

    clear_session()

    return result