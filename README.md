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
| LSTM              | 41.24    | 80.98     | 0.0875  |
| CNN-LSTM          | 41.36    | 82.41     | 0.0551  |
| GRU               | 42.88    | 83.05     | 0.0402  |
| Random Forest     | 65.99    | 96.33     | -0.2912 |

Best model: Linear Regression
Optimization performed using Random Search over 6 configurations.
