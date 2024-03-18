# Car Insurance Data Analysis

This Python script is designed for analyzing and predicting outcomes based on car insurance data. It employs various libraries such as Pandas, Scikit-learn, NumPy, and Matplotlib to process, analyze, and model the data.

## Overview

The script processes a dataset from a CSV file named `car_insurance.csv`, handling missing values, encoding categorical variables, normalizing data, and applying machine learning models to predict outcomes. It uses logistic regression, perceptron, and K-nearest neighbors classifiers, evaluates their performance, and saves the best model using pickle for future use.

## Key Features

- **Data Loading and Inspection:** Loads car insurance data, displays its size, data types, missing values, and initial statistics.
- **Data Cleaning:** Handles missing values by replacing them with the median and addresses outliers in specific columns.
- **Data Visualization:** Generates histograms for numerical variables and a scatter matrix for selected variables to explore relationships.
- **Data Transformation:** Encodes categorical variables into numeric ones and normalizes the data for better model performance.
- **Model Training and Evaluation:** Trains logistic regression, perceptron, and KNN models, evaluates them using cross-validation, and computes various metrics like accuracy, precision, recall, and F1 score.
- **Model Persistence:** Saves and loads the best model using the pickle library.

## Usage

1. Ensure you have Python installed along with the necessary libraries: pandas, sklearn, numpy, matplotlib.
2. Place the `car_insurance.csv` file in the same directory as the script.
3. Run the script using a Python interpreter. The script will process the data, train models, and output performance metrics.
4. The trained logistic regression model will be saved and can be loaded for future predictions.

## Code Structure

- The script starts by importing the required libraries.
- It then loads the data, performs initial data analysis, and cleans the data by handling missing values and outliers.
- Data visualization is performed to understand the distribution and relationships between variables.
- Categorical variables are encoded, and the data is normalized.
- Machine learning models are instantiated, trained, and evaluated.
- The best-performing model (logistic regression in this context) is saved to disk.
- Finally, the saved model can be loaded for future predictions.
