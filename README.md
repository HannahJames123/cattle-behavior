# cattle-behavior
Advancing Standardized Cattle Behavior Classification with a Random Forest Model Evaluated Across Diverse Datasets

# Cattle Behavior Classification (Random Forest)

This repository contains a Python implementation for classifying cattle behavior based on accelerometer data using a Random Forest model.

## Files
- `cattle_behavior_model.py`: Main script for preprocessing, training, and evaluating the model.
- Your dataset (CSV) should be named `data.csv` with columns: `acc_x`, `acc_y`, `acc_z`, `behavior`.

## Steps
1. Segments time-series data into 0.5s windows
2. Extracts statistical features (mean, std, skew, kurtosis)
3. Handles missing values and balances classes using SMOTE
4. Trains a Random Forest model and evaluates accuracy, F1, precision, recall
5. Plots a confusion matrix

## Requirements
- Python 3.x
- pandas, numpy, scikit-learn, imbalanced-learn, seaborn, matplotlib

