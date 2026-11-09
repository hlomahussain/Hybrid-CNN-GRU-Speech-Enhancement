"""
Network definitions.

hybrid   : Conv1D encoder (learned spectral features) + 3 stacked GRUs  -> proposed model
cnn_only : same encoder + two more Conv1D layers, no recurrence          -> ablation (no temporal memory)
gru_only : 3 stacked GRUs directly on the log-Mel input, no encoder      -> ablation (no learned encoder)

All three share: input = log-Mel (T, 128) in dB, output = 22 band gains in [0, 1],
the same loss, optimiser and training data. Input normalisation (mean / variance measured on
the training set) is stored inside the model, so a saved model is self-contained.
"""
if __name__ == "__main__":
    import os
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")   # hide TensorFlow start-up messages
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    import _env  # noqa: F401  (Run button: uses the project environment automatically)

import numpy as np
import tensorflow as tf
import keras
from keras import layers

import config as C


@keras.saving.register_keras_serializable(package="se")
def perceptual_loss(y_true, y_pred):
    """L = mean( (g^gamma - g_hat^gamma)^2 ),  gamma = 0.5   (RNNoise, eq. 8)"""
    g = C.LOSS_GAMMA
    y_true = tf.clip_by_value(y_true, 1e-7, 1.0)
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0)
    return tf.reduce_mean(tf.square(tf.pow(y_true, g) - tf.pow(y_pred, g)))


def _conv(x, filters):
    x = layers.Conv1D(filters, 3, padding=C.CONV_PADDING, activation="relu")(x)
    return layers.BatchNormalization()(x)


def _gru(x, units):
    return layers.GRU(units, return_sequences=True, activation=C.GRU_ACTIVATION)(x)


def build_model(kind, norm_mean, norm_var):
    inputs = layers.Input(shape=(None, C.N_MELS), name="log_mel")
    x = layers.Normalization(axis=-1, mean=norm_mean, variance=norm_var, name="input_norm")(inputs)

    if kind == "hybrid":
        x = _conv(x, 32)
        x = _conv(x, 64)
        x = _gru(x, 24)
        x = _gru(x, 48)
        x = _gru(x, 96)
    elif kind == "cnn_only":
        for f in (32, 64, 96, 128):
            x = _conv(x, f)
    elif kind == "gru_only":
        x = _gru(x, 24)
        x = _gru(x, 48)
        x = _gru(x, 96)
    else:
        raise ValueError(f"unknown model kind: {kind}")

    outputs = layers.Dense(C.NUM_BANDS, activation="sigmoid", name="band_gains")(x)
    return keras.Model(inputs, outputs, name=kind)


def load_trained(kind):
    import os
    path = os.path.join(C.MODELS_DIR, f"{kind}.keras")
    return keras.models.load_model(path, compile=False) if os.path.exists(path) else None


def count_params(model):
    trainable = int(np.sum([np.prod(w.shape) for w in model.trainable_weights]))
    total = int(np.sum([np.prod(w.shape) for w in model.weights]))
    return trainable, total


if __name__ == "__main__":
    # Run this file to see the three trained networks: layers, output shapes and parameter counts.
    for kind, name in C.MODEL_NAMES.items():
        model = load_trained(kind)
        if model is None:
            print(f"{kind}: models/{kind}.keras not found")
            continue
        print("\n" + "=" * 78)
        print(f" {name}")
        print("=" * 78)
        model.summary(line_length=78)
        trainable, _ = count_params(model)
        print(f" Trainable parameters: {trainable:,}")
    print("\nInput: 128 log-Mel values per 10 ms frame.  Output: 22 band gains in [0, 1] per frame.")
