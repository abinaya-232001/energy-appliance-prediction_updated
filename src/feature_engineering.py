"""
feature_engineering.py
───────────────────────
Creates all engineered features from the preprocessed dataframe:
- Time-based features
- Cyclical encoding
- Interaction and aggregated features
- Rolling statistics (leak-proof)
- Lagged features
- Feature set definitions (RAW and ENGINEERED)
- Min-Max scaling
- Sequence creation for recurrent models
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os
from sklearn.preprocessing import MinMaxScaler
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


# ── Feature set definitions ────────────────────────────────────────────────────

RAW_FEATURES = [
    'lights',
    'T1', 'RH_1', 'T2', 'RH_2', 'T3', 'RH_3',
    'T4', 'RH_4', 'T5', 'RH_5', 'T6', 'RH_6',
    'T7', 'RH_7', 'T8', 'RH_8', 'T9', 'RH_9',
    'T_out', 'Press_mm_hg', 'RH_out', 'Windspeed',
    'Visibility', 'Tdewpoint',
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
    'is_weekend', 'NSM',
    'avg_indoor_temp', 'avg_indoor_RH',
    'temp_diff', 'temp_humidity_out'
]
"""
RAW_FEATURES — used by sequence models (LSTM, GRU, CNN-LSTM).
Excludes lag and rolling stats deliberately: recurrent models
learn temporal dependencies from the lookback window itself.
Adding explicit lags here would be redundant and could cause
leakage if the window overlaps the lag lookback.
"""

ENGINEERED_FEATURES = RAW_FEATURES + [
    'rolling_mean_6',  'rolling_mean_18', 'rolling_mean_36',
    'rolling_std_6',   'rolling_std_18',  'rolling_std_36',
    'lag_1', 'lag_3', 'lag_6', 'lag_12', 'lag_18', 'lag_36'
]
"""
ENGINEERED_FEATURES — used by flat models (Linear Regression, Random Forest).
Flat models have no memory of past steps, so explicit temporal summaries
(lags, rolling stats) are required to capture any time-series structure.
"""

TARGET      = 'Appliances_capped'
LOOKBACK    = 36    # 36 × 10 min = 6 hours; justified by ACF analysis


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time-based and cyclical features.

    Features added:
        month       — calendar month (seasonal signal)
        is_weekend  — 1 if Sat/Sun, else 0 (usage pattern differs)
        NSM         — seconds since midnight (continuous within-day signal)
        hour_sin/cos — hour encoded as unit circle (avoids 23→0 discontinuity)
        dow_sin/cos  — day-of-week encoded cyclically

    Why cyclical encoding:
        Raw integer hour (0–23) creates a false gap between 23 and 0.
        Sine/cosine encoding wraps hours onto a circle so the model
        sees 23:00 and 00:00 as adjacent, not maximally different.

    Args:
        df (pd.DataFrame): Dataframe with 'hour' and 'day_of_week' already extracted.

    Returns:
        pd.DataFrame: Dataframe with time features added.
    """
    df['month']      = df['date'].dt.month
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['NSM']        = df['hour'] * 3600 + df['date'].dt.minute * 60

    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow_sin']  = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos']  = np.cos(2 * np.pi * df['day_of_week'] / 7)

    return df


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add interaction and aggregated sensor features.

    Features added:
        avg_indoor_temp   — mean of T1–T6; reduces 6 correlated sensors to one signal
        avg_indoor_RH     — mean of RH_1–RH_6; same rationale
        temp_diff         — T1 − T_out; proxy for heating/cooling demand
        temp_humidity_out — T_out × RH_out; outdoor heat-index proxy

    Args:
        df (pd.DataFrame): Dataframe with raw sensor columns.

    Returns:
        pd.DataFrame: Dataframe with interaction features added.
    """
    df['avg_indoor_temp']   = df[['T1','T2','T3','T4','T5','T6']].mean(axis=1)
    df['avg_indoor_RH']     = df[['RH_1','RH_2','RH_3','RH_4','RH_5','RH_6']].mean(axis=1)
    df['temp_diff']         = df['T1'] - df['T_out']
    df['temp_humidity_out'] = df['T_out'] * df['RH_out']

    return df


def add_rolling_features(df: pd.DataFrame,
                          windows: list = [6, 18, 36],
                          target_col: str = 'Appliances') -> pd.DataFrame:
    """Add rolling mean and std features with leak-proof shifting.

    The target series is shifted by 1 step BEFORE applying the rolling
    window. This ensures the rolling statistic for row t is computed
    from rows t-window to t-1, never including row t itself.
    Without the shift, the rolling window would include the current
    value, which constitutes data leakage.

    Args:
        df         (pd.DataFrame): Input dataframe.
        windows    (list):         Rolling window sizes in timesteps.
        target_col (str):          Column to compute rolling stats on.

    Returns:
        pd.DataFrame: Dataframe with rolling features added.
    """
    shifted = df[target_col].shift(1)
    for w in windows:
        df[f'rolling_mean_{w}'] = shifted.rolling(window=w, min_periods=1).mean()
        df[f'rolling_std_{w}']  = shifted.rolling(window=w, min_periods=1).std().fillna(0)

    return df


def add_lag_features(df: pd.DataFrame,
                     lags: list = [1, 3, 6, 12, 18, 36],
                     target_col: str = 'Appliances') -> pd.DataFrame:
    """Add lagged target values as features.

    Lag periods are justified by ACF analysis — significant
    autocorrelation at lags 1, 3, 6, 12, 18, 36 confirms that
    past values at these intervals carry predictive signal.

    Args:
        df         (pd.DataFrame): Input dataframe.
        lags       (list):         Lag steps to create.
        target_col (str):          Column to lag.

    Returns:
        pd.DataFrame: Dataframe with lag features added.
    """
    for lag in lags:
        df[f'lag_{lag}'] = df[target_col].shift(lag)

    return df


def plot_acf_pacf(df: pd.DataFrame,
                  target_col: str = 'Appliances',
                  lags: int = 50,
                  figures_dir: str = 'reports/figures') -> None:
    """Plot ACF and PACF to justify lag selection and lookback window.

    Args:
        df          (pd.DataFrame): Dataframe containing target column.
        target_col  (str):          Column to analyse.
        lags        (int):          Number of lags to plot.
        figures_dir (str):          Directory to save plots.
    """
    os.makedirs(figures_dir, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7))
    plot_acf( df[target_col], lags=lags, ax=ax1)
    plot_pacf(df[target_col], lags=lags, ax=ax2)
    ax1.set_title('ACF — Justifies Lag Feature Selection and Lookback Window')
    ax2.set_title('PACF — Indicates Direct Autoregressive Order')
    plt.tight_layout()
    plt.savefig(f'{figures_dir}/acf_pacf.png', dpi=150)
    plt.close()
    print(f"ACF/PACF plot saved to {figures_dir}/acf_pacf.png")
    print("Interpretation: Significant autocorrelation up to lag ~36 justifies")
    print("  LOOKBACK=36 and lag features at [1, 3, 6, 12, 18, 36].")


def run_feature_engineering(df: pd.DataFrame,
                             figures_dir: str = 'reports/figures') -> pd.DataFrame:
    """Full feature engineering pipeline — single entry point.

    Runs all steps in order:
    1. Time-based features
    2. Cyclical encoding
    3. Interaction and aggregated features
    4. Rolling statistics
    5. Lagged features
    6. Drop NaN rows from lag operations

    Args:
        df          (pd.DataFrame): Preprocessed dataframe (with hour, day_of_week).
        figures_dir (str):          Directory to save ACF/PACF plots.

    Returns:
        pd.DataFrame: Dataframe with all engineered features.
    """
    pre_len = len(df)

    df = add_time_features(df)
    df = add_interaction_features(df)
    df = add_rolling_features(df)
    df = add_lag_features(df)

    # Drop NaN rows introduced by lag operations (first max_lag rows)
    df = df.dropna().reset_index(drop=True)
    print(f"Rows dropped (lag NaN): {pre_len - len(df)}")
    print(f"Shape after feature engineering: {df.shape}")
    print(f"Total columns: {len(df.columns)}")

    # ACF/PACF — placed after feature engineering to inform lookback choice
    plot_acf_pacf(df, figures_dir=figures_dir)

    return df


def verify_features(df: pd.DataFrame) -> None:
    """Confirm all defined feature columns exist in the dataframe.

    Args:
        df (pd.DataFrame): Fully engineered dataframe.

    Raises:
        AssertionError: If any expected features are missing.
    """
    missing_raw = [f for f in RAW_FEATURES        if f not in df.columns]
    missing_eng = [f for f in ENGINEERED_FEATURES if f not in df.columns]

    if missing_raw:
        raise AssertionError(f"Missing RAW features: {missing_raw}")
    if missing_eng:
        raise AssertionError(f"Missing ENGINEERED features: {missing_eng}")

    print(f"RAW_FEATURES ({len(RAW_FEATURES)}):        all present ✓")
    print(f"ENGINEERED_FEATURES ({len(ENGINEERED_FEATURES)}): all present ✓")


def scale_features(train_df: pd.DataFrame,
                   test_df: pd.DataFrame,
                   models_dir: str = 'models') -> tuple:
    """Apply Min-Max scaling to features and target.

    Scaling method: Min-Max (range [0, 1])
    Justification:
    - Neural networks converge faster when inputs are bounded.
    - Min-Max preserves the original distribution shape.
    - The right-skewed target does not meet Gaussian assumptions,
      so Standardisation is less appropriate here.

    CRITICAL — no data leakage:
    All scalers are fit ONLY on training data, then applied
    (transform only) to test data. Fitting on the full dataset
    would allow test statistics to influence training.

    Scalers saved to disk for inference use.

    Args:
        train_df   (pd.DataFrame): Training split.
        test_df    (pd.DataFrame): Test split.
        models_dir (str):          Directory to save fitted scalers.

    Returns:
        tuple: (X_train_raw_sc, X_test_raw_sc,
                X_train_eng_sc, X_test_eng_sc,
                y_train_sc, y_test_sc,
                scaler_X_raw, scaler_X_eng, scaler_y)
    """
    os.makedirs(models_dir, exist_ok=True)

    # Target
    y_train = train_df[TARGET].values.reshape(-1, 1)
    y_test  = test_df[TARGET].values.reshape(-1, 1)
    scaler_y   = MinMaxScaler()
    y_train_sc = scaler_y.fit_transform(y_train)
    y_test_sc  = scaler_y.transform(y_test)

    # Raw features — for sequence models
    scaler_X_raw   = MinMaxScaler()
    X_train_raw_sc = scaler_X_raw.fit_transform(train_df[RAW_FEATURES].values)
    X_test_raw_sc  = scaler_X_raw.transform(test_df[RAW_FEATURES].values)

    # Engineered features — for flat models
    scaler_X_eng   = MinMaxScaler()
    X_train_eng_sc = scaler_X_eng.fit_transform(train_df[ENGINEERED_FEATURES].values)
    X_test_eng_sc  = scaler_X_eng.transform(test_df[ENGINEERED_FEATURES].values)

    # Save scalers
    joblib.dump(scaler_X_raw, f'{models_dir}/scaler_X_raw.pkl')
    joblib.dump(scaler_X_eng, f'{models_dir}/scaler_X_eng.pkl')
    joblib.dump(scaler_y,     f'{models_dir}/scaler_y.pkl')

    print("Scaled array shapes:")
    print(f"  X_train_raw_sc: {X_train_raw_sc.shape}")
    print(f"  X_test_raw_sc:  {X_test_raw_sc.shape}")
    print(f"  X_train_eng_sc: {X_train_eng_sc.shape}")
    print(f"  X_test_eng_sc:  {X_test_eng_sc.shape}")
    print(f"  y_train_sc:     {y_train_sc.shape}")
    print(f"  y_test_sc:      {y_test_sc.shape}")
    print(f"Scalers saved to {models_dir}/")

    return (X_train_raw_sc, X_test_raw_sc,
            X_train_eng_sc, X_test_eng_sc,
            y_train_sc, y_test_sc,
            scaler_X_raw, scaler_X_eng, scaler_y)


def create_sequences(X: np.ndarray,
                     y: np.ndarray,
                     lookback: int = LOOKBACK) -> tuple:
    """Convert flat 2D arrays to overlapping 3D sequences for recurrent models.

    Each output sample contains `lookback` consecutive timesteps
    of all features, paired with the target at the next timestep.
    Temporal order is strictly preserved — no shuffling.

    Args:
        X        (np.ndarray): Scaled feature array [n_samples, n_features].
        y        (np.ndarray): Scaled target array  [n_samples, 1].
        lookback (int):        Number of past timesteps per sequence.

    Returns:
        tuple: (Xs, ys)
            Xs shape: [n_samples - lookback, lookback, n_features]
            ys shape: [n_samples - lookback, 1]
    """
    Xs, ys = [], []
    for i in range(lookback, len(X)):
        Xs.append(X[i - lookback : i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)


if __name__ == '__main__':
    from data_preprocessing import run_preprocessing

    df, train_df, test_df = run_preprocessing('data/energy_data_set.csv')
    df = run_feature_engineering(df)

    # Re-split after feature engineering (NaN rows dropped)
    from data_preprocessing import split_train_test
    train_df, test_df = split_train_test(df)

    verify_features(df)

    (X_train_raw_sc, X_test_raw_sc,
     X_train_eng_sc, X_test_eng_sc,
     y_train_sc, y_test_sc,
     scaler_X_raw, scaler_X_eng, scaler_y) = scale_features(train_df, test_df)

    X_tr_seq, y_tr_seq = create_sequences(X_train_raw_sc, y_train_sc)
    X_te_seq, y_te_seq = create_sequences(X_test_raw_sc,  y_test_sc)

    print(f"\nTrain sequences: {X_tr_seq.shape}")
    print(f"Test sequences:  {X_te_seq.shape}")