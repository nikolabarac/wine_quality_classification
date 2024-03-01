# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 12:43:32 2024

@author: Nikola
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv('D:/Python_e/wine_quality/winequality-red.csv')

column_list = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
               'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
               'pH', 'sulphates', 'alcohol', 'quality', 'class']

# Set up subplots
fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(15, 13))
fig.suptitle('Histograms for Each Column', y=1.02)

# Flatten the 2D array of subplots into a 1D array
axes = axes.flatten()

# Plot histograms for each column
for i, column in enumerate(column_list):
    axes[i].hist(df[column], bins=20, color='skyblue', edgecolor='black')
    axes[i].set_title(column)
    axes[i].set_xlabel('Value')
    axes[i].set_ylabel('Frequency')

# Adjust layout
plt.tight_layout()
plt.show()

df.info()

from sklearn.model_selection import train_test_split

# Assuming 'class' is your target column
target_column = 'class'

# Separate data into features (X) and target variable (y)
X = df.drop(target_column, axis=1)  # Assuming 'class' is the target column
y = df[target_column]

# Split the data into train and test sets while maintaining class balance
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Check the class distribution in train and test sets
print("Train Set Class Distribution:")
print(y_train.value_counts())

print("\nTest Set Class Distribution:")
print(y_test.value_counts())

# Specify the file paths for saving train and test sets
train_path = 'D:/Python_e/wine_quality_classification/train_set.csv'
test_path = 'D:/Python_e/wine_quality_classification/test_set.csv'

# Save X_train and y_train to a CSV file for the training set
train_set = pd.concat([X_train, y_train], axis=1)
train_set.to_csv(train_path, index=False)

# Save X_test and y_test to a CSV file for the testing set
test_set = pd.concat([X_test, y_test], axis=1)
test_set.to_csv(test_path, index=False)

print(f"Train set saved to: {train_path}")
print(f"Test set saved to: {test_path}")