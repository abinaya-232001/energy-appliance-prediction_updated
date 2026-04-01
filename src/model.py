"""
model.py
────────
Defines all model architectures:
- build_lstm()     — Stacked LSTM with BatchNorm and Dropout
- build_gru()      — Stacked GRU with BatchNorm and Dropout
- build_cnn_lstm() — CNN feature extractor + LSTM temporal model

All models use Huber loss and Adam optimiser.
Training callbacks (EarlyStopping, ReduceLROnPlateau) are defined here.
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    LSTM, GRU, Dense, Dropout,
    Conv1D, MaxPooling1D, Input,
    BatchNormalization
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam


def get_callbacks(patience_stop: int = 15,
                  patience_lr: int = 7,
                  min_lr: float = 1e-6) -> list:
    """Return standard training callbacks.

    Callbacks:
        EarlyStopping:
            - Monitors val_loss; restores best weights automatically.
            - Stops training when val_loss shows no improvement for
              `patience_stop` epochs, preventing wasted compute.

        ReduceLROnPlateau:
            - Reduces learning rate by factor 0.3 when val_loss
              stalls for `patience_lr` epochs.
            - Allows the optimiser to fine-tune convergence after
              large initial steps.

    Args:
        patience_stop (int):   Epochs without improvement before stopping.
        patience_lr   (int):   Epochs without improvement before LR reduction.
        min_lr        (float): Minimum learning rate floor.

    Returns:
        list: [EarlyStopping, ReduceLROnPlateau]
    """
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=patience_stop,
        restore_best_weights=True,
        verbose=1
    )
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.3,
        patience=patience_lr,
        min_lr=min_lr,
        verbose=1
    )
    return [early_stop, reduce_lr]


def build_lstm(lookback: int, n_features: int,
               lr: float = 0.001) -> tf.keras.Model:
    """Build a stacked LSTM model for energy consumption regression.

    Architecture:
        LSTM(128) → BatchNorm → Dropout(0.3)
        LSTM(64)  → BatchNorm → Dropout(0.2)
        LSTM(32)  → Dropout(0.2)
        Dense(64, ReLU) → Dropout(0.1)
        Dense(32, ReLU)
        Dense(1)  [linear — regression output]

    Design choices:
        - Three stacked LSTM layers: deeper stacks capture more abstract
          temporal representations at increasing levels of abstraction.
        - BatchNormalization between layers: stabilises activations and
          significantly speeds convergence in deep recurrent networks.
        - Recurrent dropout (0.1): regularisation applied within LSTM
          cells at each timestep; differs from standard dropout which
          operates between layers only.
        - Huber loss: combines MSE stability for small errors with MAE
          robustness for large errors — well-suited for the right-skewed
          energy distribution with residual outliers after capping.
        - Adam (lr=0.001): adaptive learning rate handles sparse and
          noisy gradients; standard choice for recurrent models.

    Args:
        lookback   (int):   Sequence length (timesteps).
        n_features (int):   Number of input features per timestep.
        lr         (float): Initial learning rate for Adam.

    Returns:
        tf.keras.Model: Compiled LSTM model.
    """
    model = Sequential([
        LSTM(128, return_sequences=True,
             input_shape=(lookback, n_features),
             recurrent_dropout=0.1),
        BatchNormalization(),
        Dropout(0.3),
        LSTM(64, return_sequences=True, recurrent_dropout=0.1),
        BatchNormalization(),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.1),
        Dense(32, activation='relu'),
        Dense(1)
    ], name='LSTM_Model')

    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss='huber',
        metrics=['mae']
    )
    return model


def build_gru(lookback: int, n_features: int,
              lr: float = 0.001) -> tf.keras.Model:
    """Build a stacked GRU model for energy consumption regression.

    Architecture:
        GRU(128) → BatchNorm → Dropout(0.3)
        GRU(64)  → BatchNorm → Dropout(0.2)
        Dense(64, ReLU) → Dropout(0.1)
        Dense(32, ReLU)
        Dense(1)  [linear — regression output]

    Comparison to LSTM:
        GRU merges the forget and input gates into a single update gate,
        reducing parameter count. This typically leads to faster training
        with comparable predictive performance. Including GRU alongside
        LSTM allows empirical comparison to determine which recurrent
        architecture captures energy patterns better on this dataset.

    Args:
        lookback   (int):   Sequence length (timesteps).
        n_features (int):   Number of input features per timestep.
        lr         (float): Initial learning rate for Adam.

    Returns:
        tf.keras.Model: Compiled GRU model.
    """
    model = Sequential([
        GRU(128, return_sequences=True,
            input_shape=(lookback, n_features),
            recurrent_dropout=0.1),
        BatchNormalization(),
        Dropout(0.3),
        GRU(64, return_sequences=False, recurrent_dropout=0.1),
        BatchNormalization(),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.1),
        Dense(32, activation='relu'),
        Dense(1)
    ], name='GRU_Model')

    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss='huber',
        metrics=['mae']
    )
    return model


def build_cnn_lstm(lookback: int, n_features: int,
                   lr: float = 0.001) -> tf.keras.Model:
    """Build a CNN-LSTM hybrid model for energy consumption regression.

    Architecture:
        Conv1D(128, kernel=3, ReLU, same)
        Conv1D(64,  kernel=3, ReLU, same)
        MaxPooling1D(2)
        Dropout(0.2)
        LSTM(64, return_sequences=True) → Dropout(0.2)
        LSTM(32, return_sequences=False) → Dropout(0.2)
        Dense(64, ReLU)
        Dense(32, ReLU)
        Dense(1)  [linear — regression output]

    Design rationale:
        CNN layers function as local feature extractors — they identify
        short-term patterns (e.g., hour-level spikes) within the sequence
        window. MaxPooling downsamples the sequence length by 2, reducing
        computational cost and providing light regularisation. The LSTM
        layers then model long-range dependencies on this compressed,
        enriched representation. This hierarchy (local → global) can
        outperform pure LSTM or GRU when both short-term patterns and
        long-term dependencies are present in the data.

    Args:
        lookback   (int):   Sequence length (timesteps).
        n_features (int):   Number of input features per timestep.
        lr         (float): Initial learning rate for Adam.

    Returns:
        tf.keras.Model: Compiled CNN-LSTM model.
    """
    inputs  = Input(shape=(lookback, n_features))
    x       = Conv1D(128, kernel_size=3, activation='relu', padding='same')(inputs)
    x       = Conv1D(64,  kernel_size=3, activation='relu', padding='same')(x)
    x       = MaxPooling1D(pool_size=2)(x)
    x       = Dropout(0.2)(x)
    x       = LSTM(64, return_sequences=True)(x)
    x       = Dropout(0.2)(x)
    x       = LSTM(32, return_sequences=False)(x)
    x       = Dropout(0.2)(x)
    x       = Dense(64, activation='relu')(x)
    x       = Dense(32, activation='relu')(x)
    outputs = Dense(1)(x)

    model = Model(inputs, outputs, name='CNN_LSTM')
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss='huber',
        metrics=['mae']
    )
    return model


if __name__ == '__main__':
    # Quick sanity check — build and summarise all three models
    LOOKBACK   = 36
    N_FEATURES = 35

    print("=" * 50)
    print("LSTM MODEL")
    build_lstm(LOOKBACK, N_FEATURES).summary()

    print("=" * 50)
    print("GRU MODEL")
    build_gru(LOOKBACK, N_FEATURES).summary()

    print("=" * 50)
    print("CNN-LSTM MODEL")
    build_cnn_lstm(LOOKBACK, N_FEATURES).summary()