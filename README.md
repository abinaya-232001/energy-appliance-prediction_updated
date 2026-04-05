# Appliance Energy Prediction Using Deep Learning

## Overview
Predicts household appliance energy consumption using multivariate
time-series deep learning models (LSTM, GRU, CNN-LSTM) on the
Appliance Energy Prediction Dataset.

## Dataset
- 19,735 records at 10-minute intervals (January-May 2016)
- Target: Appliances energy consumption (Wh)
- Source: UCI Machine Learning Repository

## Repository Structure
data/
  raw/
  processed/
notebooks/
  EDA.ipynb
src/
  data_preprocessing.py
  feature_engineering.py
  model.py
  train.py
models/
  trained_model.h5
  optimized_model.h5
reports/
  report.pdf
  figures/
requirements.txt
README.md

## Setup
pip install -r requirements.txt


## Run
Place energy_data_set.csv in working directory.
Open notebooks/EDA.ipynb and run all cells in order.
Or run: python src/train.py

## Results

| Model             | MAE (Wh) | RMSE (Wh) | R²      |
|-------------------|----------|-----------|---------|
| Linear Regression | 26.33    | 57.36     | 0.5422  |
| LSTM              | 34.79    | 80.81     | 0.0912  |
| CNN-LSTM          | 42.52    | 82.94     | 0.0428  |
| GRU               | 51.33    | 88.85     | −0.0986  |
| Random Forest     | 62.65    | 92.14     | −0.1813 |

Best model: Linear Regression
Optimization performed using Random Search over 6 configurations.
