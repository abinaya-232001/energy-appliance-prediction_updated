"""
data_preprocessing.py
─────────────────────
Handles all data loading, cleaning, outlier treatment,
and train/test splitting for the Appliance Energy Prediction dataset.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')


def load_and_sort(filepath: str) -> pd.DataFrame:
    """Load the dataset, parse dates, and sort chronologically.

    Args:
        filepath (str): Path to the raw CSV file.

    Returns:
        pd.DataFrame: Sorted dataframe with datetime index.
    """
    df = pd.read_csv(filepath, parse_dates=['date'])
    df = df.sort_values('date').reset_index(drop=True)
    print(f"Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Date range: {df['date'].min()}  →  {df['date'].max()}")
    return df


def drop_noise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove intentionally injected random noise columns rv1 and rv2.

    These columns are documented noise variables with near-zero correlation
    to the target. Retaining them risks the model learning spurious patterns.

    Args:
        df (pd.DataFrame): Raw dataframe.

    Returns:
        pd.DataFrame: Dataframe without rv1 and rv2.
    """
    print(f"rv1 == rv2: {(df['rv1'] == df['rv2']).all()}")
    print(f"rv1 correlation with Appliances: {df['rv1'].corr(df['Appliances']):.4f}")
    print(f"rv2 correlation with Appliances: {df['rv2'].corr(df['Appliances']):.4f}")
    df = df.drop(columns=['rv1', 'rv2'])
    print(f"Shape after noise removal: {df.shape}")
    return df


def check_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Detect and handle missing values.

    Strategy: forward-fill (ffill) then backward-fill (bfill).
    Rationale: Preserves temporal continuity in time-series data.
    Mean imputation is avoided as it introduces statistical artefacts
    inconsistent with the temporal structure of the series.
    Columns exceeding 50% missing are dropped entirely.

    Args:
        df (pd.DataFrame): Input dataframe.

    Returns:
        pd.DataFrame: Dataframe with missing values handled.
    """
    missing_counts = df.isnull().sum()
    missing_pct    = (missing_counts / len(df)) * 100

    report = pd.DataFrame({
        'Missing Count': missing_counts,
        'Missing %':     missing_pct.round(2)
    }).sort_values('Missing Count', ascending=False)

    has_missing = report[report['Missing Count'] > 0]

    if len(has_missing) > 0:
        print("Columns with missing values:")
        print(has_missing)

        # Drop features with >50% missing — imputation unreliable at that scale
        cols_to_drop = has_missing[has_missing['Missing %'] > 50].index.tolist()
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
            print(f"Dropped (>50% missing): {cols_to_drop}")

        df = df.fillna(method='ffill').fillna(method='bfill')
        print(f"Missing values after imputation: {df.isnull().sum().sum()}")
    else:
        print(f"Total missing values: {df.isnull().sum().sum()}")
        print("No missing values detected. No imputation required.")

    return df


def detect_and_cap_outliers(df: pd.DataFrame,
                             target_col: str = 'Appliances',
                             cap_percentile: float = 0.99,
                             figures_dir: str = 'reports/figures') -> pd.DataFrame:
    """Detect outliers via IQR method and cap the target at the 99th percentile.

    Rationale for capping over removal:
    - Many high-energy spikes are real events (HVAC, oven), not errors.
    - Removing rows destroys temporal continuity in a 10-minute series.
    - Capping preserves row count and temporal order while dampening
      extreme values that would otherwise dominate gradient updates.

    Args:
        df              (pd.DataFrame): Input dataframe.
        target_col      (str):          Name of the target column.
        cap_percentile  (float):        Percentile for the upper cap (default 0.99).
        figures_dir     (str):          Directory to save output plots.

    Returns:
        pd.DataFrame: Dataframe with a new '<target_col>_capped' column.
    """
    Q1  = df[target_col].quantile(0.25)
    Q3  = df[target_col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = df[(df[target_col] < lower) | (df[target_col] > upper)]
    print(f"IQR bounds: [{lower:.1f}, {upper:.1f}] Wh")
    print(f"Outliers detected: {len(outliers)} ({100 * len(outliers) / len(df):.2f}%)")

    cap_val = df[target_col].quantile(cap_percentile)
    capped_col = f"{target_col}_capped"
    df[capped_col] = df[target_col].clip(upper=cap_val)
    n_capped = (df[target_col] > cap_val).sum()
    print(f"Cap at {cap_percentile*100:.0f}th percentile: {cap_val:.1f} Wh")
    print(f"Records affected: {n_capped} ({100 * n_capped / len(df):.2f}%)")

    # Visualise
    os.makedirs(figures_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].boxplot(df[target_col])
    axes[0].set_title('Before Capping')
    axes[0].set_ylabel('Energy (Wh)')
    axes[1].boxplot(df[capped_col])
    axes[1].set_title(f'After {cap_percentile*100:.0f}th Percentile Capping')
    axes[1].set_ylabel('Energy (Wh)')
    plt.tight_layout()
    plt.savefig(f'{figures_dir}/outlier_treatment.png', dpi=150)
    plt.close()
    print(f"Outlier plot saved to {figures_dir}/outlier_treatment.png")

    return df


def split_train_test(df: pd.DataFrame,
                     train_ratio: float = 0.80) -> tuple:
    """Split the dataset chronologically into train and test sets.

    IMPORTANT: The dataset is NEVER shuffled. Shuffling a time series
    allows future data to leak into the training set, producing
    unrealistically optimistic test metrics.

    Args:
        df          (pd.DataFrame): Full preprocessed dataframe.
        train_ratio (float):        Proportion for training (default 0.80).

    Returns:
        tuple: (train_df, test_df)
    """
    split_idx = int(len(df) * train_ratio)
    train_df  = df.iloc[:split_idx].copy()
    test_df   = df.iloc[split_idx:].copy()

    print(f"Train: {len(train_df)} rows | {train_df['date'].iloc[0]} → {train_df['date'].iloc[-1]}")
    print(f"Test:  {len(test_df)} rows  | {test_df['date'].iloc[0]} → {test_df['date'].iloc[-1]}")
    print(f"Split ratio: {len(train_df)/len(df):.0%} train / {len(test_df)/len(df):.0%} test")

    return train_df, test_df


def run_preprocessing(filepath: str,
                      figures_dir: str = 'reports/figures') -> tuple:
    """Full preprocessing pipeline — single entry point.

    Runs all steps in order:
    1. Load and sort
    2. Drop noise columns
    3. Handle missing values
    4. Detect and cap outliers
    5. Extract basic time components for EDA
    6. Chronological train/test split

    Args:
        filepath    (str): Path to the raw CSV.
        figures_dir (str): Directory to save output plots.

    Returns:
        tuple: (df, train_df, test_df)
    """
    df = load_and_sort(filepath)
    df = drop_noise_columns(df)
    df = check_missing_values(df)

    # Basic time components — needed before feature engineering and EDA
    df['hour']        = df['date'].dt.hour
    df['day_of_week'] = df['date'].dt.dayofweek

    df = detect_and_cap_outliers(df, figures_dir=figures_dir)

    train_df, test_df = split_train_test(df)

    return df, train_df, test_df


if __name__ == '__main__':
    df, train_df, test_df = run_preprocessing('data/energy_data_set.csv')
    print(f"\nFinal shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")