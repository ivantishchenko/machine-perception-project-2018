#!/usr/bin/env python3
"""For a given prediction csv calculate Kaggle private metric."""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + '/../src/'))
from util.gaze import mean_angular_error  # noqa

# Parse arguments
parser = argparse.ArgumentParser(description='Calculate Kaggle private test subset MSE metric.')
parser.add_argument('predictions_csv', type=str,
                    help='CSV file containing correctly formatted predictions.')
args = parser.parse_args()

# Read predictions CSV
pred_csv_path = args.predictions_csv
pred_csv = pd.read_csv(pred_csv_path)

# Read ground-truth CSV
true_csv_path = os.path.dirname(__file__) + '/../datasets/kaggle_solution.csv'
true_csv = pd.read_csv(true_csv_path)


def _mse(pred, true):
    return np.mean((pred - true) ** 2)

# Calculate metric for public test set
public_pred = pred_csv.loc[true_csv['Usage'] == 'Public'][['pitch', 'yaw']].as_matrix()
public_true = true_csv.loc[true_csv['Usage'] == 'Public'][['pitch', 'yaw']].as_matrix()
print('Public MSE: %f' % _mse(public_pred, public_true))
print('       ang: %f' % mean_angular_error(public_pred, public_true))

# Calculate metric for private test set
private_pred = pred_csv.loc[true_csv['Usage'] == 'Private'][['pitch', 'yaw']].as_matrix()
private_true = true_csv.loc[true_csv['Usage'] == 'Private'][['pitch', 'yaw']].as_matrix()
print('Private MSE: %f' % _mse(private_pred, private_true))
print('        ang: %f' % mean_angular_error(private_pred, private_true))
