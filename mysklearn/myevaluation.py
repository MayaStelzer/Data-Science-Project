"""
Programmer: Maura Sweeney
Class: CPSC 322, Fall 2025
Programming Assignment #7
11/25/25

Description:
Evaluation functions for machine learning including train/test splitting, k-fold cross validation, bootstrap sampling, confusion matrices, and performance metrics like accuracy, precision, recall, and F1 score
"""

from mysklearn import myutils
import numpy as np
from tabulate import tabulate

def train_test_split(X, y, test_size=0.33, random_state=None, shuffle=True):
    """Split dataset into train and test sets based on a test set size.

    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X)
            The shape of y is n_samples
        test_size(float or int): float for proportion of dataset to be in test set (e.g. 0.33 for a 2:1 split)
            or int for absolute number of instances to be in test set (e.g. 5 for 5 instances in test set)
        random_state(int): integer used for seeding a random number generator for reproducible results
            Use random_state to seed your random number generator
                you can use the math module or use numpy for your generator
                choose one and consistently use that generator throughout your code
        shuffle(bool): whether or not to randomize the order of the instances before splitting
            Shuffle the rows in X and y before splitting and be sure to maintain the parallel order of X and y!!

    Returns:
        X_train(list of list of obj): The list of training samples
        X_test(list of list of obj): The list of testing samples
        y_train(list of obj): The list of target y values for training (parallel to X_train)
        y_test(list of obj): The list of target y values for testing (parallel to X_test)

    Note:
        Loosely based on sklearn's train_test_split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    """
    if random_state is not None:
        # use given value as random seed
        rand_num_gen = np.random.RandomState(random_state)
    else:
        rand_num_gen = np.random

    size = len(X)

    if isinstance(test_size, float):
        test_count = int(np.ceil(size*test_size))
    else:
        # int
        test_count = test_size

    indexes = list(range(size))

    if shuffle:
        rand_num_gen.shuffle(indexes)
        # take test from front
        test_idx = indexes[:test_count]
        train_idx = indexes[test_count:]
    else:
        # take test from end
        test_idx = indexes[-test_count:]
        train_idx = indexes[:-test_count]

    X_train = [X[i] for i in train_idx]
    X_test = [X[i] for i in test_idx]
    y_train = [y[i] for i in train_idx]
    y_test = [y[i] for i in test_idx]

    return X_train, X_test, y_train, y_test

def kfold_split(X, n_splits=5, random_state=None, shuffle=False):
    """Split dataset into cross validation folds.

    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        n_splits(int): Number of folds.
        random_state(int): integer used for seeding a random number generator for reproducible results
        shuffle(bool): whether or not to randomize the order of the instances before creating folds

    Returns:
        folds(list of 2-item tuples): The list of folds where each fold is defined as a 2-item tuple
            The first item in the tuple is the list of training set indices for the fold
            The second item in the tuple is the list of testing set indices for the fold

    Notes:
        The first n_samples % n_splits folds have size n_samples // n_splits + 1,
            other folds have size n_samples // n_splits, where n_samples is the number of samples
            (e.g. 11 samples and 4 splits, the sizes of the 4 folds are 3, 3, 3, 2 samples)
        Loosely based on sklearn's KFold split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
    """
    if random_state is not None:
        # use given value as random seed
        rand_num_gen = np.random.RandomState(random_state)
    else:
        rand_num_gen = np.random

    size = len(X)
    indexes = list(range(size))

    if shuffle:
        rand_num_gen.shuffle(indexes)

    # try to divide equally 
    fold_size = [size // n_splits] * n_splits
    for i in range(size % n_splits):
        fold_size[i] += 1

    # -> into folds
    cut_off_points = []
    start = 0
    for fsize in fold_size:
        cut_off_points.append(indexes[start:(start + fsize)])
        start += fsize

    # build list of (train, test) index pairs
    folds = []
    for i in range(n_splits):
        test = cut_off_points[i]
        train = [idx for j in range(n_splits) if j != i for idx in cut_off_points[j]]
        folds.append((train, test))

    return folds

# BONUS function
def stratified_kfold_split(X, y, n_splits=5, random_state=None, shuffle=False):
    """Split dataset into stratified cross validation folds.

    Args:
        X(list of list of obj): The list of instances (samples).
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X).
            The shape of y is n_samples
        n_splits(int): Number of folds.
        random_state(int): integer used for seeding a random number generator for reproducible results
        shuffle(bool): whether or not to randomize the order of the instances before creating folds

    Returns:
        folds(list of 2-item tuples): The list of folds where each fold is defined as a 2-item tuple
            The first item in the tuple is the list of training set indices for the fold
            The second item in the tuple is the list of testing set indices for the fold

    Notes:
        Loosely based on sklearn's StratifiedKFold split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold
    """
    return [] # TODO: (BONUS) fix this

def bootstrap_sample(X, y=None, n_samples=None, random_state=None):
    """Split dataset into bootstrapped training set and out of bag test set.

    Args:
        X(list of list of obj): The list of samples
        y(list of obj): The target y values (parallel to X)
            Default is None (in this case, the calling code only wants to sample X)
        n_samples(int): Number of samples to generate. If left to None (default) this is automatically
            set to the first dimension of X.
        random_state(int): integer used for seeding a random number generator for reproducible results

    Returns:
        X_sample(list of list of obj): The list of samples
        X_out_of_bag(list of list of obj): The list of "out of bag" samples (e.g. left-over samples)
        y_sample(list of obj): The list of target y values sampled (parallel to X_sample)
            None if y is None
        y_out_of_bag(list of obj): The list of target y values "out of bag" (parallel to X_out_of_bag)
            None if y is None
    Notes:
        Loosely based on sklearn's resample():
            https://scikit-learn.org/stable/modules/generated/sklearn.utils.resample.html
        Sample indexes of X with replacement, then build X_sample and X_out_of_bag
            as lists of instances using sampled indexes (use same indexes to build
            y_sample and y_out_of_bag)
    """
    if random_state is not None:
        # use given value as random seed
        rand_num_gen = np.random.RandomState(random_state)
    else:
        rand_num_gen = np.random

    if n_samples is None:
        n_samples = len(X)

    indexes = list(range(len(X)))
    sample_index = list(rand_num_gen.choice(indexes, size=n_samples, replace=True))
    out_of_bag_index = [i for i in indexes if i not in sample_index]

    X_sample = [X[i] for i in sample_index]
    X_out_of_bag = [X[i] for i in out_of_bag_index]

    if y is None:
        return X_sample, X_out_of_bag, None, None
    else:
        y_sample = [y[i] for i in sample_index]
        y_out_of_bag = [y[i] for i in out_of_bag_index]
        return X_sample, X_out_of_bag, y_sample, y_out_of_bag


def confusion_matrix(y_true, y_pred, labels):
    """Compute confusion matrix to evaluate the accuracy of a classification.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of str): The list of all possible target y labels used to index the matrix

    Returns:
        matrix(list of list of int): Confusion matrix whose i-th row and j-th column entry
            indicates the number of samples with true label being i-th class
            and predicted label being j-th class

    Notes:
        Loosely based on sklearn's confusion_matrix():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    """
    index_label = {lab: idx for idx, lab in enumerate(labels)}

    # matrix of zeros
    size = len(labels)
    matrix = [[0 for _ in range(size)] for _ in range(size)]

    for i in range(len(y_true)):
        true_label = y_true[i]
        pred_label = y_pred[i]
        row = index_label[true_label]
        col = index_label[pred_label]
        matrix[row][col] += 1

    return matrix

def accuracy_score(y_true, y_pred, normalize=True):
    """Compute the classification prediction accuracy score.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        normalize(bool): If False, return the number of correctly classified samples.
            Otherwise, return the fraction of correctly classified samples.

    Returns:
        score(float): If normalize == True, return the fraction of correctly classified samples (float),
            else returns the number of correctly classified samples (int).

    Notes:
        Loosely based on sklearn's accuracy_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score
    """
    correct = 0
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            correct+=1
    if normalize:
        return correct/len(y_true)
    else:
        return correct


def binary_precision_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the precision (for binary classification). The precision is the ratio tp / (tp + fp)
        where tp is the number of true positives and fp the number of false positives.
        The precision is intuitively the ability of the classifier not to label as
        positive a sample that is negative. The best value is 1 and the worst value is 0.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels

    Returns:
        precision(float): Precision of the positive class

    Notes:
        Loosely based on sklearn's precision_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html
    """
    # Handle default values for labels and pos_label
    if labels is None:
        labels = sorted(list(set(y_true)))
    if pos_label is None:
        pos_label = labels[0]

    # Calculate TP and FP
    tp = 0  # true positives
    fp = 0  # false positives

    for i in range(len(y_true)):
        if y_pred[i] == pos_label:
            if y_true[i] == pos_label:
                tp += 1
            else:
                fp += 1

    # Calculate precision
    if tp + fp == 0:
        return 0.0
    return tp / (tp + fp)

def binary_recall_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the recall (for binary classification). The recall is the ratio tp / (tp + fn) where tp is
        the number of true positives and fn the number of false negatives.
        The recall is intuitively the ability of the classifier to find all the positive samples.
        The best value is 1 and the worst value is 0.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels

    Returns:
        recall(float): Recall of the positive class

    Notes:
        Loosely based on sklearn's recall_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html
    """
    # Handle default values for labels and pos_label
    if labels is None:
        labels = sorted(list(set(y_true)))
    if pos_label is None:
        pos_label = labels[0]

    # Calculate TP and FN
    tp = 0  # true positives
    fn = 0  # false negatives

    for i in range(len(y_true)):
        if y_true[i] == pos_label:
            if y_pred[i] == pos_label:
                tp += 1
            else:
                fn += 1

    # Calculate recall
    if tp + fn == 0:
        return 0.0
    return tp / (tp + fn)

def binary_f1_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the F1 score (for binary classification), also known as balanced F-score or F-measure.
        The F1 score can be interpreted as a harmonic mean of the precision and recall,
        where an F1 score reaches its best value at 1 and worst score at 0.
        The relative contribution of precision and recall to the F1 score are equal.
        The formula for the F1 score is: F1 = 2 * (precision * recall) / (precision + recall)

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels

    Returns:
        f1(float): F1 score of the positive class

    Notes:
        Loosely based on sklearn's f1_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
    """
    # Calculate precision and recall
    precision = binary_precision_score(y_true, y_pred, labels=labels, pos_label=pos_label)
    recall = binary_recall_score(y_true, y_pred, labels=labels, pos_label=pos_label)

    # Calculate F1 score
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

def classification_report(y_true, y_pred, labels=None, output_dict=False):
    """ text report and a dictionary showing main classification metrics

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        output_dict(bool): If True, return output as dict instead of a str

    Returns:
        report(str or dict): Text summary of the precision, recall, F1 score for each class.
            Dictionary returned if output_dict is True. Dictionary has the following structure:
                {'label 1': {'precision':0.5,
                            'recall':1.0,
                            'f1-score':0.67,
                            'support':1},
                'label 2': { ... },
                ...
                }
            The reported averages include macro average (averaging the unweighted mean per label) and
            weighted average (averaging the support-weighted mean per label).
            Micro average (averaging the total true positives, false negatives and false positives)
            multi-class with a subset of classes, because it corresponds to accuracy otherwise
            and would be the same for all metrics.

    Notes:
        Loosely based on sklearn's classification_report():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
    """
    # default labels
    if labels is None:
        labels = sorted(list(set(y_true)))

    report_dict = {}

    # metrics for each label
    total_support = 0
    sum_precision = 0.0
    sum_recall = 0.0
    sum_f1 = 0.0
    weighted_precision = 0.0
    weighted_recall = 0.0
    weighted_f1 = 0.0

    for label in labels:
        # number of true instances
        support = y_true.count(label)
        total_support += support

        # precision, recall, f1 for class
        precision = binary_precision_score(y_true, y_pred, labels=labels, pos_label=label)
        recall = binary_recall_score(y_true, y_pred, labels=labels, pos_label=label)
        f1 = binary_f1_score(y_true, y_pred, labels=labels, pos_label=label)

        # store in dictionary
        report_dict[str(label)] = {
            'precision': precision,
            'recall': recall,
            'f1-score': f1,
            'support': support
        }

        sum_precision += precision
        sum_recall += recall
        sum_f1 += f1
        weighted_precision += precision * support
        weighted_recall += recall * support
        weighted_f1 += f1 * support

    # macro average
    num_labels = len(labels)
    report_dict['macro avg'] = {
        'precision': sum_precision / num_labels if num_labels > 0 else 0.0,
        'recall': sum_recall / num_labels if num_labels > 0 else 0.0,
        'f1-score': sum_f1 / num_labels if num_labels > 0 else 0.0,
        'support': total_support
    }

    # weighted average 
    report_dict['weighted avg'] = {
        'precision': weighted_precision / total_support if total_support > 0 else 0.0,
        'recall': weighted_recall / total_support if total_support > 0 else 0.0,
        'f1-score': weighted_f1 / total_support if total_support > 0 else 0.0,
        'support': total_support
    }

    if output_dict:
        return report_dict

    headers = ['', 'precision', 'recall', 'f1-score', 'support']
    table_data = []

    for label in labels:
        label_str = str(label)
        row = [
            label_str,
            f"{report_dict[label_str]['precision']:.2f}",
            f"{report_dict[label_str]['recall']:.2f}",
            f"{report_dict[label_str]['f1-score']:.2f}",
            report_dict[label_str]['support']
        ]
        table_data.append(row)
    table_data.append(['', '', '', '', ''])

    # macro average
    table_data.append([
        'macro avg',
        f"{report_dict['macro avg']['precision']:.2f}",
        f"{report_dict['macro avg']['recall']:.2f}",
        f"{report_dict['macro avg']['f1-score']:.2f}",
        report_dict['macro avg']['support']
    ])

    # weighted average
    table_data.append([
        'weighted avg',
        f"{report_dict['weighted avg']['precision']:.2f}",
        f"{report_dict['weighted avg']['recall']:.2f}",
        f"{report_dict['weighted avg']['f1-score']:.2f}",
        report_dict['weighted avg']['support']
    ])
    table_str = tabulate(table_data, headers=headers, tablefmt='simple')

    return table_str

