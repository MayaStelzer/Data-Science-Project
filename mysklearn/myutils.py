"""
Programmer: Maura Sweeney
Class: CPSC 322, Fall 2025
Programming Assignment #6
11/12/25

Description:
Utility functions for loading, cleaning, joining, and preparing datasets
"""

from mysklearn.mypytable import MyPyTable
import numpy as np
import mysklearn.myevaluation as myevaluation
import mysklearn.myclassifiers as myclassifiers
from tabulate import tabulate

def load_data(mpg_infile, prices_infile):
    """Loads the auto MPG and prices datasets

    Parameters:
        mpg_infile (str): path to the auto-mpg.txt file
        prices_infile (str): path to the auto-prices.txt file

    Returns:
        tuple: (mpg_table, prices_table) as MyPyTable objects
    """
    mpg_table = MyPyTable().load_from_file(mpg_infile)
    prices_table = MyPyTable().load_from_file(prices_infile)
    return mpg_table, prices_table
    
def handle_duplicates(table, key_cols, outfile):
    """Finds and removes duplicate rows from a table

    Parameters:
        table (MyPyTable): table to check for duplicates
        key_cols (list of str): column names to use as row keys
        outfile (str): path to save the clean dataset to

    Returns:
        list of int: list of duplicate row indexes removed
    """
    
    dup_indices = table.find_duplicates(key_cols)
    # if there are duplicates, remove them
    if dup_indices:
        table.drop_rows(dup_indices)
    table.save_to_file(outfile)
    return dup_indices

def join_tables(mpg_table, price_table, outfile):
    """Performs a full outer join between MPG and Price tables

    Parameters:
        mpg_table (MyPyTable): table containing MPG values
        price_table (MyPyTable): table containing MSRP values
        outfile (str): path to save the joined dataset

    Returns:
        MyPyTable: resulting full outer joined table
    """
    
    joined_table = mpg_table.perform_full_outer_join(price_table, ["car name", "model year"])
    joined_table.save_to_file(outfile)
    return joined_table

def count_frequencies(values):
    """Returns a dictionary of value frequencies"""
    freqs = {}
    for v in values:
        freqs[v] = freqs.get(v, 0) + 1
    return freqs

def summary_stats(table, columns, outfile):
    """Computes and saves summary statistics for numeric columns

    Parameters:
        table (MyPyTable): table to summarize
        columns (list of str): numeric column names to compute stats for
        outfile (str): path to save summary statistics table

    Returns:
        MyPyTable: summary statistics table
    """
    
    stats = table.compute_summary_statistics(columns)
    stats.save_to_file(outfile)
    return stats


def handle_missing_values(table, method, outfile):
    """Handles missing values using removal or replacement

    Parameters:
        table (MyPyTable): table with missing values
        method (str): 'remove' to drop rows or 'replace' to fill with averages
        outfile (str): path to save the modified dataset

    Returns:
        MyPyTable: new table with missing values handled
    """
    new_table = MyPyTable(table.column_names[:], [row[:] for row in table.data])

    if method == "remove":
        new_table.remove_rows_with_missing_values()
    elif method == "replace":
        for col in new_table.column_names:
            new_table.replace_missing_values_with_column_average(col)

    new_table.save_to_file(outfile)
    return new_table

def mpg_to_doe_rating(mpg):
    if mpg >= 45: return 10
    elif 37 <= mpg <= 44: return 9
    elif 31 <= mpg <= 36: return 8
    elif 27 <= mpg <= 30: return 7
    elif 24 <= mpg <= 26: return 6
    elif 20 <= mpg <= 23: return 5
    elif 17 <= mpg <= 19: return 4
    elif 15 <= mpg <= 16: return 3
    elif mpg == 14: return 2
    else: return 1


def random_subsample(X, y, k=10, test_size=0.33, random_state=None):

    knn_accuracies = []
    dummy_accuracies = []

    for i in range(k):
        # random subsampling
        X_train, X_test, y_train, y_test = myevaluation.train_test_split(X, y, test_size=test_size, random_state=(None if random_state is None else random_state + i), shuffle=True)

        # KNN classifier
        knn = myclassifiers.MyKNeighborsClassifier()
        knn.fit(X_train, y_train)
        knn_pred = knn.predict(X_test)
        knn_acc = myevaluation.accuracy_score(y_test, knn_pred)
        knn_accuracies.append(knn_acc)

        # dummy classifier
        dummy = myclassifiers.MyDummyClassifier()
        dummy.fit(X_train, y_train)
        dummy_pred = dummy.predict(X_test)
        dummy_acc = myevaluation.accuracy_score(y_test, dummy_pred)
        dummy_accuracies.append(dummy_acc)

    # compute average accuracy and error rate
    knn_accuracy = sum(knn_accuracies) / k
    dummy_accuracy = sum(dummy_accuracies) / k

    knn_error = 1 - knn_accuracy
    dummy_error = 1 - dummy_accuracy

    # formatted output
    print("===========================================")
    print("STEP 1: Predictive Accuracy")
    print("===========================================")
    print(f"Random Subsample (k={k}, {1-test_size:.0f}:{test_size:.0f} Train/Test)")
    print(f"k Nearest Neighbors Classifier: accuracy = {knn_accuracy:.2f}, error rate = {knn_error:.2f}")
    print(f"Dummy Classifier: accuracy = {dummy_accuracy:.2f}, error rate = {dummy_error:.2f}")


def evaluate_kfold(X, y, n_splits=10):
    folds = myevaluation.kfold_split(X, n_splits=n_splits, shuffle=False)

    knn = myclassifiers.MyKNeighborsClassifier()
    dummy = myclassifiers.MyDummyClassifier()

    knn_scores = []
    dummy_scores = []

    for train_idx, test_idx in folds:
        if len(test_idx) == 0:
            continue

        X_train = [X[i] for i in train_idx]
        y_train = [y[i] for i in train_idx]
        X_test = [X[i] for i in test_idx]
        y_test = [y[i] for i in test_idx]

        knn.fit(X_train, y_train)
        y_pred_knn = knn.predict(X_test)
        knn_scores.append(myevaluation.accuracy_score(y_test, y_pred_knn))

    dummy.fit(X_train, y_train)
    y_pred_dummy = dummy.predict(X_test)
    dummy_scores.append(myevaluation.accuracy_score(y_test, y_pred_dummy))


    knn_acc = sum(knn_scores) / len(knn_scores)
    dummy_acc = sum(dummy_scores) / len(dummy_scores)

    print("===========================================")
    print("STEP 2: Predictive Accuracy")
    print("===========================================")
    print("10-Fold Cross Validation")
    print(f"k Nearest Neighbors Classifier: accuracy = {knn_acc:.2f}, error rate = {1-knn_acc:.2f}")
    print(f"Dummy Classifier: accuracy = {dummy_acc:.2f}, error rate = {1-dummy_acc:.2f}")

    return folds

def evaluate_bootstrap(X, y, k=10):
    knn_scores = []
    dummy_scores = []

    for i in range(k):
        X_sample, X_out_of_bag, y_sample, y_out_of_bag = myevaluation.bootstrap_sample(X, y, random_state=i)

        knn = myclassifiers.MyKNeighborsClassifier()
        knn.fit(X_sample, y_sample)
        knn_pred = knn.predict(X_out_of_bag)
        knn_scores.append(myevaluation.accuracy_score(y_out_of_bag, knn_pred))

        dummy = myclassifiers.MyDummyClassifier()
        dummy.fit(X_sample, y_sample)
        dummy_pred = dummy.predict(X_out_of_bag)
        dummy_scores.append(myevaluation.accuracy_score(y_out_of_bag, dummy_pred))

    knn_acc = sum(knn_scores) / k
    dummy_acc = sum(dummy_scores) / k

    print("===========================================")
    print("STEP 3: Predictive Accuracy")
    print("===========================================")
    print("k=10 Bootstrap Method")
    print(f"k Nearest Neighbors Classifier: accuracy = {knn_acc:.2f}, error rate = {1-knn_acc:.2f}")
    print(f"Dummy Classifier: accuracy = {dummy_acc:.2f}, error rate = {1-dummy_acc:.2f}")


def confusion_matrices(X, y, folds):
    labels = sorted(list(set(y)))

    knn_cm = [[0 for _ in labels] for _ in labels]
    dummy_cm = [[0 for _ in labels] for _ in labels]

    for train_idx, test_idx in folds:
        X_train = [X[i] for i in train_idx]
        y_train = [y[i] for i in train_idx]
        X_test = [X[i] for i in test_idx]
        y_test = [y[i] for i in test_idx]

        knn = myclassifiers.MyKNeighborsClassifier()
        knn.fit(X_train, y_train)
        knn_pred = knn.predict(X_test)
        cm = myevaluation.confusion_matrix(y_test, knn_pred, labels)
        for i in range(len(labels)):
            for j in range(len(labels)):
                knn_cm[i][j] += cm[i][j]

        dummy = myclassifiers.MyDummyClassifier()
        dummy.fit(X_train, y_train)
        dummy_pred = dummy.predict(X_test)
        cm = myevaluation.confusion_matrix(y_test, dummy_pred, labels)
        for i in range(len(labels)):
            for j in range(len(labels)):
                dummy_cm[i][j] += cm[i][j]

    print("===========================================")
    print("STEP 4: Confusion Matrices")
    print("===========================================")
    print("kNN Confusion Matrix:")
    print(tabulate(knn_cm, headers=labels, showindex=labels))
    print("\nDummy Confusion Matrix:")
    print(tabulate(dummy_cm, headers=labels, showindex=labels))

def print_confusion_matrix(cm, labels, title):
    print(f"\n{title}:")
    print("-" * 40)
    
    # Header
    print(f"{'':>15}", end="")
    for label in labels:
        print(f"{label:>10}", end="")
    print(" | Total")
    print("-" * 40)
    
    # Rows
    for i, label in enumerate(labels):
        print(f"{label:>15}", end="")
        row_sum = sum(cm[i])
        for j in range(len(labels)):
            print(f"{cm[i][j]:>10}", end="")
        print(f" | {row_sum}")
    
    print("-" * 40)
    print(f"{'Total':>15}", end="")
    for j in range(len(labels)):
        col_sum = sum(cm[i][j] for i in range(len(labels)))
        print(f"{col_sum:>10}", end="")
    print(f" | {sum(sum(row) for row in cm)}")