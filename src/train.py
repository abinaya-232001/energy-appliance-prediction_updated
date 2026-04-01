"""
train.py
────────
Full training, evaluation, and optimisation pipeline.
Orchestrates:
  1. Preprocessing
  2. Feature engineering
  3. Scaling and sequence creation
  4. Baseline models (Linear Regression, Random Forest)
  5. Deep learning models (LSTM, GRU, CNN-LSTM)
  6. Evaluation and comparison
  7. Visualisation
  8. Model saving

Run from the project root:
    python src/train.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import warnings
import tensorflow as tf
warnings.filterwarnings('ignore')

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Local modules
from data_preprocessing  import run_preprocessing, split_train_test
from feature_engineering import (
    run_feature_engineering, verify_features,
    scale_features, create_sequences,
    RAW_FEATURES, ENGINEERED_FEATURES, TARGET, LOOKBACK
)
from model import build_lstm, build_gru, build_cnn_lstm, get_callbacks


# ── Reproducibility ────────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ── Directories ────────────────────────────────────────────────────────────────
FIGURES_DIR = 'reports/figures'
MODELS_DIR  = 'models'
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR,  exist_ok=True)


# ── Evaluation helper ──────────────────────────────────────────────────────────

all_results = []

def evaluate_model(name: str,
                   y_true: np.ndarray,
                   y_pred: np.ndarray) -> tuple:
    """Compute MAE, RMSE, R² in original Wh units and record results.

    Args:
        name   (str):        Model name.
        y_true (np.ndarray): Actual values in Wh.
        y_pred (np.ndarray): Predicted values in Wh.

    Returns:
        tuple: (mae, rmse, r2)
    """
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    print(f"\n{name}:")
    print(f"  MAE:  {mae:.2f} Wh")
    print(f"  RMSE: {rmse:.2f} Wh")
    print(f"  R²:   {r2:.4f}")
    all_results.append({'Model': name, 'MAE': mae, 'RMSE': rmse, 'R2': r2})
    return mae, rmse, r2


# ── Training loss plot ─────────────────────────────────────────────────────────

def plot_training_loss(history, model_name: str) -> None:
    """Plot and save training vs validation loss curves.

    Args:
        history    (keras History): Returned by model.fit().
        model_name (str):           Used in title and filename.
    """
    plt.figure(figsize=(10, 4))
    plt.plot(history.history['loss'],     label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{model_name}: Training vs Validation Loss (Huber)')
    plt.xlabel('Epoch')
    plt.ylabel('Huber Loss')
    plt.legend()
    plt.tight_layout()
    fname = model_name.lower().replace('-', '_').replace(' ', '_')
    plt.savefig(f'{FIGURES_DIR}/{fname}_loss.png', dpi=150)
    plt.close()
    print(f"  Loss plot saved: {FIGURES_DIR}/{fname}_loss.png")


# ── Baseline models ────────────────────────────────────────────────────────────

def train_baselines(X_train_eng_sc: np.ndarray,
                    X_test_eng_sc: np.ndarray,
                    y_train_sc: np.ndarray,
                    y_test_sc: np.ndarray,
                    scaler_y,
                    lookback: int = LOOKBACK) -> tuple:
    """Train Linear Regression and Random Forest baseline models.

    Baselines use ENGINEERED_FEATURES (includes lags and rolling stats)
    because flat models have no memory of past steps and require explicit
    temporal summaries.

    Baseline predictions are trimmed by `lookback` rows at the front
    so they align with sequence model outputs (which start `lookback`
    steps into the test set) for fair comparison.

    Args:
        X_train_eng_sc (np.ndarray): Scaled engineered train features.
        X_test_eng_sc  (np.ndarray): Scaled engineered test features.
        y_train_sc     (np.ndarray): Scaled train target.
        y_test_sc      (np.ndarray): Scaled test target.
        scaler_y:                    Fitted target scaler.
        lookback       (int):        Steps to trim for alignment.

    Returns:
        tuple: (y_test_aligned, rf_model)
            y_test_aligned — actual test values trimmed to sequence length
            rf_model       — fitted RandomForestRegressor (reused for importance)
    """
    print("\n" + "=" * 50)
    print("BASELINE MODELS")
    print("=" * 50)

    y_test_actual  = scaler_y.inverse_transform(y_test_sc).ravel()
    y_test_aligned = y_test_actual[lookback:]

    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train_eng_sc, y_train_sc.ravel())
    lr_pred    = scaler_y.inverse_transform(
                    lr.predict(X_test_eng_sc).reshape(-1, 1)).ravel()
    lr_aligned = lr_pred[lookback:]
    evaluate_model("Linear Regression", y_test_aligned, lr_aligned)

    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=-1)
    rf.fit(X_train_eng_sc, y_train_sc.ravel())
    rf_pred    = scaler_y.inverse_transform(
                    rf.predict(X_test_eng_sc).reshape(-1, 1)).ravel()
    rf_aligned = rf_pred[lookback:]
    evaluate_model("Random Forest", y_test_aligned, rf_aligned)

    # Feature importance plot
    importances = pd.Series(
        rf.feature_importances_,
        index=ENGINEERED_FEATURES
    ).sort_values(ascending=False)

    plt.figure(figsize=(10, 8))
    importances.head(20).plot(kind='barh', color='steelblue', edgecolor='white')
    plt.title('Top 20 Feature Importances (Random Forest)')
    plt.xlabel('Importance Score')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/feature_importance.png', dpi=150)
    plt.close()
    print(f"  Feature importance plot saved: {FIGURES_DIR}/feature_importance.png")

    return y_test_aligned, rf


# ── Deep learning training ─────────────────────────────────────────────────────

def train_deep_learning_models(X_tr_seq: np.ndarray,
                                y_tr_seq: np.ndarray,
                                X_te_seq: np.ndarray,
                                y_te_seq: np.ndarray,
                                scaler_y,
                                n_features: int,
                                lookback: int = LOOKBACK,
                                epochs: int = 150,
                                batch_size: int = 32,
                                val_split: float = 0.15) -> tuple:
    """Train LSTM, GRU, and CNN-LSTM models and evaluate on the test set.

    All three models use identical training configuration for fair comparison:
    - Same optimizer (Adam, lr=0.001)
    - Same loss (Huber)
    - Same callbacks (EarlyStopping, ReduceLROnPlateau)
    - Same batch size, epoch limit, and validation split

    Args:
        X_tr_seq   (np.ndarray): Train sequences [samples, lookback, features].
        y_tr_seq   (np.ndarray): Train targets.
        X_te_seq   (np.ndarray): Test sequences.
        y_te_seq   (np.ndarray): Test targets.
        scaler_y:                Fitted target scaler for inverse transform.
        n_features (int):        Number of input features per timestep.
        lookback   (int):        Sequence length.
        epochs     (int):        Max training epochs.
        batch_size (int):        Samples per gradient update.
        val_split  (float):      Fraction of train data for validation monitoring.

    Returns:
        tuple: (lstm_model, gru_model, cnn_lstm_model,
                actual_lstm, pred_lstm,
                actual_gru,  pred_gru,
                actual_cnn,  pred_cnn)
    """
    callbacks  = get_callbacks()
    fit_kwargs = dict(
        epochs=epochs,
        batch_size=batch_size,
        validation_split=val_split,
        callbacks=callbacks,
        verbose=1
    )

    # ── LSTM ──────────────────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("LSTM MODEL")
    print("=" * 50)
    lstm_model = build_lstm(lookback, n_features)
    lstm_model.summary()
    history_lstm = lstm_model.fit(X_tr_seq, y_tr_seq, **fit_kwargs)
    plot_training_loss(history_lstm, 'LSTM')

    pred_lstm_sc = lstm_model.predict(X_te_seq)
    pred_lstm    = scaler_y.inverse_transform(pred_lstm_sc).ravel()
    actual_lstm  = scaler_y.inverse_transform(y_te_seq.reshape(-1, 1)).ravel()
    evaluate_model("LSTM", actual_lstm, pred_lstm)

    # ── GRU ───────────────────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("GRU MODEL")
    print("=" * 50)
    gru_model = build_gru(lookback, n_features)
    gru_model.summary()
    history_gru = gru_model.fit(X_tr_seq, y_tr_seq, **fit_kwargs)
    plot_training_loss(history_gru, 'GRU')

    pred_gru_sc = gru_model.predict(X_te_seq)
    pred_gru    = scaler_y.inverse_transform(pred_gru_sc).ravel()
    actual_gru  = scaler_y.inverse_transform(y_te_seq.reshape(-1, 1)).ravel()
    evaluate_model("GRU", actual_gru, pred_gru)

    # ── CNN-LSTM ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("CNN-LSTM MODEL")
    print("=" * 50)
    cnn_lstm_model = build_cnn_lstm(lookback, n_features)
    cnn_lstm_model.summary()
    history_cnn = cnn_lstm_model.fit(X_tr_seq, y_tr_seq, **fit_kwargs)
    plot_training_loss(history_cnn, 'CNN-LSTM')

    pred_cnn_sc = cnn_lstm_model.predict(X_te_seq)
    pred_cnn    = scaler_y.inverse_transform(pred_cnn_sc).ravel()
    actual_cnn  = scaler_y.inverse_transform(y_te_seq.reshape(-1, 1)).ravel()
    evaluate_model("CNN-LSTM", actual_cnn, pred_cnn)

    return (lstm_model, gru_model, cnn_lstm_model,
            actual_lstm, pred_lstm,
            actual_gru,  pred_gru,
            actual_cnn,  pred_cnn)


# ── Evaluation visualisation ───────────────────────────────────────────────────

def plot_evaluation(actual_lstm, pred_lstm,
                    actual_gru,  pred_gru,
                    actual_cnn,  pred_cnn,
                    results_df:  pd.DataFrame) -> None:
    """Generate and save all evaluation plots.

    Plots produced:
    1. Predicted vs Actual (first 400 test samples, all 3 DL models)
    2. Residual plot (LSTM — Actual minus Predicted)
    3. Metrics comparison bar charts (MAE, RMSE, R² across all 5 models)

    Args:
        actual_* / pred_* (np.ndarray): Model outputs in original Wh units.
        results_df (pd.DataFrame):      Comparison table from all_results.
    """
    # Predicted vs Actual
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    for ax, (name, actual, pred) in zip(axes, [
        ("LSTM",     actual_lstm, pred_lstm),
        ("GRU",      actual_gru,  pred_gru),
        ("CNN-LSTM", actual_cnn,  pred_cnn),
    ]):
        ax.plot(actual[:400], label='Actual',
                linewidth=1.0, color='steelblue')
        ax.plot(pred[:400],   label='Predicted',
                linewidth=1.0, linestyle='--', alpha=0.85, color='crimson')
        ax.set_title(f'{name}: Predicted vs Actual Energy Consumption')
        ax.set_ylabel('Energy (Wh)')
        ax.set_xlabel('Test Step')
        ax.legend()
    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/predicted_vs_actual.png', dpi=150)
    plt.close()

    # Residuals (LSTM)
    residuals = actual_lstm - pred_lstm
    plt.figure(figsize=(14, 4))
    plt.plot(residuals[:400], color='darkorange', alpha=0.75, linewidth=0.8)
    plt.axhline(0, linestyle='--', color='black', linewidth=1.0)
    plt.title('Residuals: LSTM — (Actual − Predicted)')
    plt.ylabel('Residual (Wh)')
    plt.xlabel('Test Step')
    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/residuals.png', dpi=150)
    plt.close()
    print(f"Residuals — Mean: {residuals.mean():.2f} Wh  |  Std: {residuals.std():.2f} Wh")

    # Metrics comparison
    colors = ['#4472C4', '#ED7D31', '#70AD47', '#FFC000', '#C00000']
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (metric, title) in zip(axes, [
        ('MAE',  'MAE (Wh) — Lower is Better'),
        ('RMSE', 'RMSE (Wh) — Lower is Better'),
        ('R2',   'R² — Higher is Better'),
    ]):
        results_df.plot(
            x='Model', y=metric, kind='bar',
            ax=ax, color=colors[:len(results_df)], legend=False
        )
        ax.set_title(title)
        ax.set_xlabel('')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha='right')
    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/metrics_comparison.png', dpi=150)
    plt.close()

    print(f"All evaluation plots saved to {FIGURES_DIR}/")


# ── Model saving ───────────────────────────────────────────────────────────────

def save_best_model(results_df: pd.DataFrame,
                    lstm_model, gru_model, cnn_lstm_model,
                    models_dir: str = MODELS_DIR) -> None:
    """Save the best-performing deep learning model based on MAE.

    HDF5 format attempted first; falls back to TensorFlow SavedModel.
    Scalers are saved separately in scale_features() — this function
    saves only the model weights and architecture.

    Args:
        results_df     (pd.DataFrame): Sorted comparison table.
        lstm_model:                    Trained LSTM.
        gru_model:                     Trained GRU.
        cnn_lstm_model:                Trained CNN-LSTM.
        models_dir     (str):          Directory to save model.
    """
    model_map = {
        'LSTM':     lstm_model,
        'GRU':      gru_model,
        'CNN-LSTM': cnn_lstm_model
    }
    best_name  = results_df.iloc[0]['Model']
    best_model = model_map.get(best_name, lstm_model)

    print(f"\nSaving best model: {best_name}")
    try:
        best_model.save(f'{models_dir}/trained_model.h5')
        print(f"  Saved: {models_dir}/trained_model.h5")
    except Exception as e:
        print(f"  HDF5 failed ({e}) — saving as SavedModel.")
        best_model.save(f'{models_dir}/trained_model_savedmodel')
        print(f"  Saved: {models_dir}/trained_model_savedmodel/")

    print("\nArtifact verification:")
    for path in [f'{models_dir}/trained_model.h5',
                 f'{models_dir}/scaler_X_raw.pkl',
                 f'{models_dir}/scaler_X_eng.pkl',
                 f'{models_dir}/scaler_y.pkl']:
        status = "✓ exists" if os.path.exists(path) else "✗ MISSING"
        print(f"  {path:<40} {status}")


# ── Main pipeline ──────────────────────────────────────────────────────────────

def main():
    """Run the full training pipeline end-to-end."""

    print("=" * 60)
    print("APPLIANCE ENERGY PREDICTION — FULL TRAINING PIPELINE")
    print("=" * 60)

    # ── Step 1: Preprocessing ─────────────────────────────────────────────────
    print("\n[1/6] PREPROCESSING")
    df, _, _ = run_preprocessing('data/energy_data_set.csv',
                                  figures_dir=FIGURES_DIR)

    # ── Step 2: Feature engineering ───────────────────────────────────────────
    print("\n[2/6] FEATURE ENGINEERING")
    df = run_feature_engineering(df, figures_dir=FIGURES_DIR)

    # Re-split after feature engineering (NaN rows dropped)
    train_df, test_df = split_train_test(df)
    verify_features(df)

    # ── Step 3: Scaling and sequences ─────────────────────────────────────────
    print("\n[3/6] SCALING AND SEQUENCE CREATION")
    (X_train_raw_sc, X_test_raw_sc,
     X_train_eng_sc, X_test_eng_sc,
     y_train_sc, y_test_sc,
     scaler_X_raw, scaler_X_eng, scaler_y) = scale_features(
        train_df, test_df, models_dir=MODELS_DIR
    )

    X_tr_seq, y_tr_seq = create_sequences(X_train_raw_sc, y_train_sc)
    X_te_seq, y_te_seq = create_sequences(X_test_raw_sc,  y_test_sc)
    n_features = X_tr_seq.shape[2]

    print(f"Train sequences: {X_tr_seq.shape}")
    print(f"Test sequences:  {X_te_seq.shape}")

    # ── Step 4: Baseline models ───────────────────────────────────────────────
    print("\n[4/6] BASELINE MODELS")
    y_test_aligned, rf_model = train_baselines(
        X_train_eng_sc, X_test_eng_sc,
        y_train_sc, y_test_sc,
        scaler_y
    )

    # ── Step 5: Deep learning models ──────────────────────────────────────────
    print("\n[5/6] DEEP LEARNING MODELS")
    (lstm_model, gru_model, cnn_lstm_model,
     actual_lstm, pred_lstm,
     actual_gru,  pred_gru,
     actual_cnn,  pred_cnn) = train_deep_learning_models(
        X_tr_seq, y_tr_seq,
        X_te_seq, y_te_seq,
        scaler_y, n_features
    )

    # ── Step 6: Evaluation and saving ─────────────────────────────────────────
    print("\n[6/6] EVALUATION, VISUALISATION AND SAVING")

    results_df = pd.DataFrame(all_results).sort_values('MAE').reset_index(drop=True)

    print("\n" + "=" * 60)
    print("FINAL MODEL COMPARISON")
    print("=" * 60)
    print(f"{'Model':<22} {'MAE (Wh)':>10} {'RMSE (Wh)':>10} {'R²':>8}")
    print("-" * 55)
    for _, row in results_df.iterrows():
        print(f"{row['Model']:<22} {row['MAE']:>10.2f}"
              f" {row['RMSE']:>10.2f} {row['R2']:>8.4f}")

    best = results_df.iloc[0]
    print(f"\n✓ Best model: {best['Model']}"
          f"  |  MAE: {best['MAE']:.2f} Wh"
          f"  |  RMSE: {best['RMSE']:.2f} Wh"
          f"  |  R²: {best['R2']:.4f}")

    plot_evaluation(
        actual_lstm, pred_lstm,
        actual_gru,  pred_gru,
        actual_cnn,  pred_cnn,
        results_df
    )

    save_best_model(results_df, lstm_model, gru_model, cnn_lstm_model)

    print("\n✓ FULL PIPELINE COMPLETE.")


if __name__ == '__main__':
    main()