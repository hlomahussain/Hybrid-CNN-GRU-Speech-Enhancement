"""
STEP 2 - train the models on identical data with identical settings.

Run:  python step2_train.py                 (trains hybrid, cnn_only, gru_only)
      python step2_train.py --model hybrid  (one model)
"""
import _env  # noqa: F401  (Run button: uses the project environment automatically)
import argparse
import os
import random

import numpy as np
import tensorflow as tf
import keras

import config as C
from models import build_model, perceptual_loss, count_params


class Batches(keras.utils.PyDataset):
    """Streams batches from the memory-mapped .npy files (training data never has to fit in RAM)."""

    def __init__(self, X, Y, batch_size, shuffle, seed=C.SEED):
        super().__init__()
        self.X, self.Y, self.bs, self.shuffle = X, Y, batch_size, shuffle
        self.rng = np.random.default_rng(seed)
        self.order = np.arange(len(X))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.X) / self.bs))

    def __getitem__(self, i):
        idx = np.sort(self.order[i * self.bs:(i + 1) * self.bs])
        return np.asarray(self.X[idx]), np.asarray(self.Y[idx])

    def on_epoch_end(self):
        if self.shuffle:
            self.rng.shuffle(self.order)


def train_one(kind, Xtr, Ytr, Xva, Yva, norm):
    random.seed(C.SEED); np.random.seed(C.SEED); tf.random.set_seed(C.SEED)
    model = build_model(kind, norm["mean"], norm["var"])
    model.compile(optimizer=keras.optimizers.Adam(C.LEARNING_RATE, clipnorm=1.0), loss=perceptual_loss)
    trainable, total = count_params(model)
    print(f"\n=== {kind}: {trainable:,} trainable parameters ===")

    os.makedirs(C.MODELS_DIR, exist_ok=True)
    os.makedirs(C.RESULTS_DIR, exist_ok=True)
    path = os.path.join(C.MODELS_DIR, f"{kind}.keras")
    callbacks = [
        keras.callbacks.ModelCheckpoint(path, monitor="val_loss", save_best_only=True),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=C.EARLY_STOP_PATIENCE, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5),
        keras.callbacks.CSVLogger(os.path.join(C.RESULTS_DIR, f"history_{kind}.csv")),
        keras.callbacks.TerminateOnNaN(),
    ]
    model.fit(Batches(Xtr, Ytr, C.BATCH_SIZE, shuffle=True), validation_data=Batches(Xva, Yva, 256, shuffle=False),
              epochs=C.EPOCHS, callbacks=callbacks, verbose=2)
    print(f"saved best model -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="all", choices=["all"] + list(C.MODEL_NAMES))
    args = ap.parse_args()

    load = lambda n: np.load(os.path.join(C.WORK_DIR, n), mmap_mode="r")
    Xtr, Ytr, Xva, Yva = load("X_train.npy"), load("Y_train.npy"), load("X_val.npy"), load("Y_val.npy")
    norm = np.load(os.path.join(C.WORK_DIR, "norm.npz"))
    print(f"train: {Xtr.shape}  val: {Xva.shape}")

    kinds = list(C.MODEL_NAMES) if args.model == "all" else [args.model]
    for kind in kinds:
        train_one(kind, Xtr, Ytr, Xva, Yva, norm)


if __name__ == "__main__":
    main()
