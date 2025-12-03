# Mushroom Classification

## Project Overview

This project classifies mushrooms as edible or poisonous using the UCI Mushroom Dataset. The project demonstrates data preprocessing, exploratory data analysis, and classification using three algorithms studied in class:

1. Decision Tree Classifier
2. Random Forest Classifier
3. K-Nearest Neighbors (KNN)

The goal is to compare classifier performance, explore feature distributions, and use test-driven development to implement and evaluate the models.

## How to Run
1. Run the Jupyter Notebook for EDA and model training and testing

```jupyter notebook MushroomClassification.ipynb```


2. Run unit tests to validate the random forest implementation:

```pytest test_myrandomforest.py```

## Project Structure
 ```
 .
├── clean_mushrooms.csv           # Cleaned dataset
├── mushrooms.csv                 # Original dataset
├── MushroomClassification.ipynb  # Notebook for EDA and model training
├── mysklearn/                    # Custom library of classifiers and utilities
│   ├── myclassifiers.py          # Decision tree, random forest, KNN implementations
│   ├── myevaluation.py           # Accuracy, confusion matrix, and metrics
│   ├── mypytable.py              # CSV loading and table utilities
│   └── myutils.py                # Helper functions
├── test_myrandomforest.py        # Unit tests for Random Forest
├── readme.md                     # Project description
└── __pycache__/                  # Python cache files
```