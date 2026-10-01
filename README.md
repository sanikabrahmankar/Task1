# Elevate Labs AI & ML Internship - Task 1: Data Cleaning & Preprocessing

## Overview
This repository contains the completion of Task 1 for the Elevate Labs AI & ML Internship. The objective was to learn how to clean and prepare raw data for machine learning models using Python, Pandas, NumPy, and Matplotlib/Seaborn.

## Steps Completed
1. Imported the dataset: Loaded the Titanic dataset and explored basic information (null values, data types).
2. Handled missing values: Dropped the 'Cabin' column, filled 'Age' with the median, and filled 'Embarked' with the mode.
3. Encoded categorical features: Applied Label Encoding to the 'Sex' column and One-Hot Encoding to the 'Embarked' column.
4. Outlier Detection: Visualized outliers in the 'Fare' column using a Seaborn boxplot and removed extreme values.
5. Feature Scaling: Standardized the 'Age' and 'Fare' numerical features using `StandardScaler`.