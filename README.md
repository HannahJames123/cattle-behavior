# Cattle Behavior Classification Across Diverse Datasets  
**Advancing Standardized Cattle Behavior Classification with a Random Forest Model**

This repository contains a Python implementation for classifying cattle behavior using accelerometer and/or gyroscope data with a Random Forest classifier. The model is designed to generalize across datasets collected with different devices and sampling rates.

## Overview

The pipeline performs the following steps:

1. **Data Segmentation**: Segments time-series data into fixed-length windows of 15 data points. This corresponds to different durations depending on the dataset's sampling rate.
2. **Feature Extraction**: Computes statistical features—mean, standard deviation, skewness, and kurtosis—across X, Y, and Z axes.
3. **Imputation and Scaling**: Handles missing values via mean imputation and standardizes feature scales.
4. **Class Balancing**: Applies SMOTE (with `k=2`) to address behavior class imbalance in the training set.
5. **Model Training**: Trains a Random Forest classifier with 100 trees and `max_features='sqrt'`.
6. **Evaluation**: Computes accuracy, F1 score, precision, recall, Gini score, and plots a confusion matrix.

## Experimental Variants

For Dataset 3 (which includes both accelerometer and gyroscope data), we evaluated three input configurations:

- **3a**: Accelerometer-only  
- **3b**: Gyroscope-only  
- **3c**: Combined accelerometer and gyroscope

This allowed comparison across sensor modalities, mirroring previous studies that report results separately for each.

## File Structure

- `80_20_final.py`: Main notebook with complete preprocessing, training, evaluation, and plots
- 'cross_validation.py': Contains implementations for 5-fold cross-validation and Leave-One-Individual-Out (LOIO) evaluation.

  ## Files
- `cattle_behavior_model.py`: Main script for preprocessing, training, and evaluating the model.
- Your dataset (CSV) should be named `data.csv` with columns: `acc_x`, `acc_y`, `acc_z`, `behavior`.

## Requirements
- Python 3.x
- pandas, numpy, scikit-learn, imbalanced-learn, seaborn, matplotlib
  
## Notes

- This implementation uses a fixed window size of 15 samples across all datasets to enable consistent model input, regardless of varying sampling rates.
- No dataset-specific tuning or hyperparameter optimization was applied to promote generalizability.
- Future extensions may include deeper tuning, recurrent models, or domain adaptation.
