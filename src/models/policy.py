import tensorflow as tf
from tensorflow import keras


class LSTMPolicy(keras.Model):
    def __init__(self, obs_dim: int, act_dim: int, lstm_units: int = 64):
        super().__init__()
        self.embed = keras.layers.Dense(128, activation="relu")
        self.lstm = keras.layers.LSTM(lstm_units, return_sequences=True, return_state=True)
        self.pi = keras.layers.Dense(act_dim)  # logits
        self.v = keras.layers.Dense(1)

    @tf.function
    def call(self, obs_seq, initial_state=None, training=False):
        x = self.embed(obs_seq)
        if initial_state is None:
            x, h, c = self.lstm(x, training=training)
        else:
            x, h, c = self.lstm(x, initial_state=initial_state, training=training)
        logits = self.pi(x)
        values = self.v(x)
        return logits, tf.squeeze(values, axis=-1), (h, c)
