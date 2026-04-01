from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.backend import clear_session

def evaluate_model(params, X_train, X_test, y_train, y_test):
    filters, kernel_size, lstm_units, dropout, lr, batch_size = params

    filters = int(filters)
    kernel_size = int(kernel_size)
    lstm_units = int(lstm_units)
    batch_size = int(batch_size)

    clear_session()

    model = Sequential()
    model.add(Conv1D(filters=filters, kernel_size=kernel_size, activation='relu',
                     input_shape=(X_train.shape[1], 1), padding='same'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(dropout))

    model.add(LSTM(lstm_units))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))

    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss='mean_squared_error',
        metrics=['mae']
    )

    history = model.fit(
        X_train, y_train,
        epochs=20,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        verbose=0
    )

    return min(history.history['val_loss'])