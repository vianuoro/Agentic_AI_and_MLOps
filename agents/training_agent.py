import tensorflow as tf
from tensorflow.keras import layers, models

class TrainingAgent:
    def build_model(self, input_shape):
        model = models.Sequential([
            layers.Input(shape=(input_shape,)),
            layers.Dense(64, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(1)  # regression output
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model

    def train_model(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        model = self.build_model(X_train.shape[1])
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=2
        )
        return model, history
