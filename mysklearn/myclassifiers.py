"""
Programmer: Maura Sweeney
Class: CPSC 322, Fall 2025
Programming Assignment #6
11/12/25

Description:
Multiple classifiers including Simple Linear Regression, k-Nearest Neighbors, Dummy, and Naive Bayes classifiers 
"""


from mysklearn import myutils
from mysklearn.mysimplelinearregressor import MySimpleLinearRegressor
import math

class MySimpleLinearRegressionClassifier:
    """Represents a simple linear regression classifier that discretizes
        predictions from a simple linear regressor (see MySimpleLinearRegressor).

    Attributes:
        discretizer(function): a function that discretizes a numeric value into
            a string label. The function's signature is func(obj) -> obj
        regressor(MySimpleLinearRegressor): the underlying regression model that
            fits a line to x and y data

    Notes:
        Terminology: instance = sample = row and attribute = feature = column
    """

    def __init__(self, discretizer, regressor=None):
        """Initializer for MySimpleLinearClassifier.

        Args:
            discretizer(function): a function that discretizes a numeric value into
                a string label. The function's signature is func(obj) -> obj
            regressor(MySimpleLinearRegressor): the underlying regression model that
                fits a line to x and y data (None if to be created in fit())
        """
        self.discretizer = discretizer
        self.regressor = regressor

    def fit(self, X_train, y_train):
        """Fits a simple linear regression line to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
        """
        # make a linear regressor if necessary
        if self.regressor is None:
            self.regressor = MySimpleLinearRegressor()

        # fit with y values
        self.regressor.fit(X_train, y_train)


    def predict(self, X_test):
        """Makes predictions for test samples in X_test by applying discretizer
            to the numeric predictions from regressor.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        # numeric predictions 
        num_pred = self.regressor.predict(X_test)

        # discretizer
        y_pred = [self.discretizer(pred) for pred in num_pred]
        return y_pred

class MyKNeighborsClassifier:
    """Represents a simple k nearest neighbors classifier.

    Attributes:
        n_neighbors(int): number of k neighbors
        X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples

    Notes:
        Loosely based on sklearn's KNeighborsClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
        Assumes data has been properly normalized before use.
    """
    def __init__(self, n_neighbors=3):
        """Initializer for MyKNeighborsClassifier.

        Args:
            n_neighbors(int): number of k neighbors
        """
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        """Fits a kNN classifier to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since kNN is a lazy learning algorithm, this method just stores X_train and y_train
        """
        self.X_train = X_train
        self.y_train = y_train

    def _euclidean_distance(self, a, b):
        """Compute Euclidean distance between two lists"""
        total = 0
        for i in range(len(a)):
            diff = float(a[i]) - float(b[i])
            total += diff * diff
        return math.sqrt(total)

    def kneighbors(self, X_test):
        """Determines the k closes neighbors of each test instance.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            distances(list of list of float): 2D list of k nearest neighbor distances
                for each instance in X_test
            neighbor_indices(list of list of int): 2D list of k nearest neighbor
                indices in X_train (parallel to distances)
        """
        neighbor_distances = []
        neighbor_indices = []

        for x_t in X_test:
            distances = []
            for i, x_train in enumerate(self.X_train):
                dist = self._euclidean_distance(x_t, x_train)
                distances.append((dist, i))

            # sort by distance and index
            distances.sort(key=lambda x: (x[0], x[1]))
            k = min(self.n_neighbors, len(distances))

            # choose k nearest distances
            k_nearest = distances[:k]
            k_distances = [d for d, _ in k_nearest]
            k_indices = [index for _, index in k_nearest]
            neighbor_distances.append(k_distances)
            neighbor_indices.append(k_indices)
        return neighbor_distances, neighbor_indices

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        _, neighbor_indices = self.kneighbors(X_test)
        y_pred = []
        for idx_list in neighbor_indices:
            # collect neighbor labels
            labels = [self.y_train[i] for i in idx_list]

            # count frequencies
            freqs = myutils.count_frequencies(labels)
            max_count = max(freqs.values())

            # find all labels tied for max_count
            tied_labels = [label for label, count in freqs.items() if count == max_count]

            # tie: first instance in training data
            chosen_label = None
            for label in self.y_train:
                if label in tied_labels:
                    chosen_label = label
                    break
            if chosen_label is None:
                chosen_label = tied_labels[0]

            y_pred.append(chosen_label)

        return y_pred
class MyDummyClassifier:
    """Represents a "dummy" classifier using the "most_frequent" strategy.
        The most_frequent strategy is a Zero-R classifier, meaning it ignores
        X_train and produces zero "rules" from it. Instead, it only uses
        y_train to see what the most frequent class label is. That is
        always the dummy classifier's prediction, regardless of X_test.

    Attributes:
        most_common_label(obj): whatever the most frequent class label in the
            y_train passed into fit()

    Notes:
        Loosely based on sklearn's DummyClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html
    """
    def __init__(self):
        """Initializer for DummyClassifier.

        """
        self.most_common_label = None

    def fit(self, X_train, y_train):
        """Fits a dummy classifier to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since Zero-R only predicts the most frequent class label, this method
                only saves the most frequent class label.
        """
        if len(y_train) == 0:
            raise ValueError("Empty y_train is invalid")
        freqs = myutils.count_frequencies(y_train)
        max_count = max(freqs.values())
        tied_labels = [label for label, count in freqs.items() if count == max_count]

        # break tie by choosing first instance
        for label in y_train:
            if label in tied_labels:
                self.most_common_label = label
                break

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        if self.most_common_label is None:
            raise ValueError("Must fit classifier before calling predict()")
        return [self.most_common_label for _ in X_test]

class MyNaiveBayesClassifier:
    """Represents a Naive Bayes classifier.

    Attributes:
        priors(YOU CHOOSE THE MOST APPROPRIATE TYPE): The prior probabilities computed for each
            label in the training set.
        conditionals(YOU CHOOSE THE MOST APPROPRIATE TYPE): The conditional probabilities computed for each
            attribute value/label pair in the training set.

    Notes:
        Loosely based on sklearn's Naive Bayes classifiers: https://scikit-learn.org/stable/modules/naive_bayes.html
        You may add additional instance attributes if you would like, just be sure to update this docstring
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyNaiveBayesClassifier.
        """
        self.priors = None
        self.conditionals = None

    def fit(self, X_train, y_train):
        """Fits a Naive Bayes classifier to X_train and y_train.

        Args:
            X_train(list of list of obj): The list of training instances (samples)
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since Naive Bayes is an eager learning algorithm, this method computes the prior probabilities
                and the conditional probabilities for the training data.
            You are free to choose the most appropriate data structures for storing the priors
                and conditionals.
        """
        # priors: P(class) = count(class) / total_count
        self.priors = {}
        class_counts = myutils.count_frequencies(y_train)
        total = len(y_train)
        for label, count in class_counts.items():
            self.priors[label] = count / total

        # conditionals: P(att=value | class)
        # {(attribute_index, attribute_value, class_label): probability}
        self.conditionals = {}

        # For each attribute index
        n_features = len(X_train[0]) if X_train else 0
        for att_idx in range(n_features):
            # each class
            for class_label in class_counts.keys():
                # all instances with this class
                class_instances = [X_train[i][att_idx] for i in range(len(X_train))
                                 if y_train[i] == class_label]

                # Count attribute values for this class
                att_value_counts = myutils.count_frequencies(class_instances)
                class_total = len(class_instances)

                # conditional probability for each attribute
                for att_value, count in att_value_counts.items():
                    key = (att_idx, att_value, class_label)
                    self.conditionals[key] = count / class_total

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_predicted = []

        for test_instance in X_test:
            # posterior probability for each class
            posteriors = {}

            for class_label in self.priors.keys():
                # prior probability
                posterior = self.priors[class_label]

                # mult by conditional probabilities for each attribute
                for att_idx, att_value in enumerate(test_instance):
                    key = (att_idx, att_value, class_label)
                    if key in self.conditionals:
                        posterior *= self.conditionals[key]
                    else:
                        posterior = 0
                        break

                posteriors[class_label] = posterior

            # Choose class with highest posterior probability
            predicted_class = max(posteriors, key=posteriors.get)
            y_predicted.append(predicted_class)

        return y_predicted
