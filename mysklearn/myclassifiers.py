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

class MyDecisionTreeClassifier:
    """Represents a decision tree classifier.

    Attributes:
        X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples
        tree(nested list): The extracted tree model.

    Notes:
        Loosely based on sklearn's DecisionTreeClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyDecisionTreeClassifier.
        """
        self.X_train = None
        self.y_train = None
        self.tree = None

    def fit(self, X_train, y_train):
        """Fits a decision tree classifier to X_train and y_train using the TDIDT
        (top down induction of decision tree) algorithm.

        Args:
            X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since TDIDT is an eager learning algorithm, this method builds a decision tree model
                from the training data.
            Build a decision tree using the nested list representation described in class.
            On a majority vote tie, choose first attribute value based on attribute domain ordering.
            Store the tree in the tree attribute.
            Use attribute indexes to construct default attribute names (e.g. "att0", "att1", ...).
        """
        self.X_train = X_train
        self.y_train = y_train

        # Create attribute domains (sorted)
        attribute_domains = {}
        for att_idx in range(len(X_train[0])):
            attribute_domains[att_idx] = sorted(list(set([row[att_idx] for row in X_train])))

        available_attributes = list(range(len(X_train[0])))

        # Build the tree - pass total dataset size as initial parent count
        self.tree = self._tdidt(X_train, y_train, available_attributes, attribute_domains, len(y_train))

    def _tdidt(self, X, y, available_attributes, attribute_domains, header_parent_count):
        """Recursive TDIDT algorithm to build decision tree.

        Args:
            X: list of instances at this node
            y: list of labels at this node
            available_attributes: list of attribute indices available for splitting
            attribute_domains: dict mapping attribute index -> sorted list of possible values
            header_parent_count: the size of the parent node BEFORE this node was created
        Returns:
            nested list representing the subtree
        """
        # Number of instances at this node BEFORE any further split
        current_partition_size = len(y)

        # Base case 1: all instances same label
        if len(set(y)) == 1:
            return ["Leaf", y[0], len(y), header_parent_count]

        # Base case 2: no attributes left
        if len(available_attributes) == 0:
            label = self._majority_vote(y)
            return ["Leaf", label, len(y), header_parent_count]

        # Select best attribute to split on
        best_attribute = self._select_attribute(X, y, available_attributes)

        # Prepare list of attributes for children
        available_attributes_copy = available_attributes.copy()
        available_attributes_copy.remove(best_attribute)

        # Create attribute node
        tree = ["Attribute", f"att{best_attribute}"]

        # Partition the data by the best attribute
        partitions = {}
        for i, instance in enumerate(X):
            value = instance[best_attribute]
            if value not in partitions:
                partitions[value] = {"X": [], "y": []}
            partitions[value]["X"].append(instance)
            partitions[value]["y"].append(y[i])

        # For each possible value in the attribute domain (sorted order)
        for value in attribute_domains[best_attribute]:
            if value in partitions:
                partition_X = partitions[value]["X"]
                partition_y = partitions[value]["y"]

                # Pass the current partition size as the parent count for children
                # This represents the size of the dataset when this split was made
                subtree = self._tdidt(
                    partition_X,
                    partition_y,
                    available_attributes_copy,
                    attribute_domains,
                    current_partition_size
                )
                tree.append(["Value", value, subtree])
            else:
                # No training instances for this value: make a 0-count leaf using majority vote of parent
                label = self._majority_vote(y)
                tree.append(["Value", value, ["Leaf", label, 0, current_partition_size]])

        return tree


    def _select_attribute(self, X, y, available_attributes):
        """Select the attribute with the highest information gain.
        
        Args:
            X: Current instances
            y: Current labels
            available_attributes: Available attribute indices
            
        Returns:
            Index of the best attribute
        """
        current_entropy = self._calculate_entropy(y)

        best_gain = None
        best_attribute = None

        for att_idx in available_attributes:
            # Partition by attribute
            partitions = {}
            for i, instance in enumerate(X):
                value = instance[att_idx]
                partitions.setdefault(value, []).append(y[i])

            # Compute weighted entropy
            weighted_entropy = 0
            for partition_y in partitions.values():
                weight = len(partition_y) / len(y)
                weighted_entropy += weight * self._calculate_entropy(partition_y)

            gain = current_entropy - weighted_entropy

            # Compare gain
            if best_gain is None:
                best_gain = gain
                best_attribute = att_idx
            else:
                # Strictly better gain
                if gain > best_gain:
                    best_gain = gain
                    best_attribute = att_idx
                # Tie-breaking rule
                elif abs(gain - best_gain) < 1e-12:
                    if att_idx < best_attribute:
                        best_attribute = att_idx

        return best_attribute

    def _calculate_entropy(self, y):
        """Calculate entropy of a label set.
        
        Args:
            y: List of labels
            
        Returns:
            Entropy value
        """
        if len(y) == 0:
            return 0
        
        # Count frequencies manually
        counts = {}
        for label in y:
            if label not in counts:
                counts[label] = 0
            counts[label] += 1
        
        entropy = 0
        for count in counts.values():
            probability = count / len(y)
            if probability > 0:
                entropy -= probability * math.log2(probability)
        
        return entropy

    def _majority_vote(self, y):
        """Return the majority class label. Ties broken alphabetically.
        
        Args:
            y: List of labels
            
        Returns:
            Most common label (alphabetically first in case of tie)
        """
        # Count frequencies manually
        counts = {}
        for label in y:
            if label not in counts:
                counts[label] = 0
            counts[label] += 1
        
        max_count = max(counts.values())
        # Get all labels with max count and sort alphabetically
        candidates = sorted([label for label, count in counts.items() if count == max_count])
        return candidates[0]

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        predictions = []
        for instance in X_test:
            prediction = self._predict_instance(instance, self.tree)
            predictions.append(prediction)
        return predictions

    def _predict_instance(self, instance, tree):
        """Recursively traverse tree to predict a single instance.
        
        Args:
            instance: Single test instance
            tree: Current tree/subtree node
            
        Returns:
            Predicted class label
        """
        # Base case: reached a leaf
        if tree[0] == "Leaf":
            return tree[1]  # Return the class label
        
        # Recursive case: internal node
        if tree[0] == "Attribute":
            attribute_name = tree[1]
            # Extract attribute index from "att#"
            att_index = int(attribute_name[3:])
            instance_value = instance[att_index]
            
            # Find the matching value branch
            for i in range(2, len(tree)):
                value_branch = tree[i]
                if value_branch[0] == "Value" and value_branch[1] == instance_value:
                    # Recursively traverse this branch
                    return self._predict_instance(instance, value_branch[2])
            
            # If no matching value found (shouldn't happen in well-formed tree)
            # Return the first leaf we can find
            for i in range(2, len(tree)):
                return self._predict_instance(instance, tree[i][2])
        
        return None

    def print_decision_rules(self, attribute_names=None, class_name="class"):
        """Prints the decision rules from the tree in the format
        "IF att == val AND ... THEN class = label", one rule on each line.

        Args:
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes
                (e.g. "att0", "att1", ...) should be used).
            class_name(str): A string to use for the class name in the decision rules
                ("class" if a string is not provided and the default name "class" should be used).
        """
        rules = []
        self._extract_rules(self.tree, [], rules, attribute_names, class_name)
        for rule in rules:
            print(rule)

    def _extract_rules(self, tree, current_path, rules, attribute_names, class_name):
        """Recursively extract decision rules from tree.
        
        Args:
            tree: Current tree/subtree
            current_path: List of (attribute, value) tuples representing current path
            rules: List to accumulate rule strings
            attribute_names: Optional list of attribute names
            class_name: Name to use for class in rules
        """
        if tree[0] == "Leaf":
            # Build rule string
            if len(current_path) == 0:
                rule = f"IF True THEN {class_name} = {tree[1]}"
            else:
                conditions = []
                for att, val in current_path:
                    if attribute_names:
                        att_name = attribute_names[att]
                    else:
                        att_name = f"att{att}"
                    conditions.append(f"{att_name} == {val}")
                rule = "IF " + " AND ".join(conditions) + f" THEN {class_name} = {tree[1]}"
            rules.append(rule)
        elif tree[0] == "Attribute":
            attribute_name = tree[1]
            att_index = int(attribute_name[3:])
            
            # Traverse each value branch
            for i in range(2, len(tree)):
                value_branch = tree[i]
                if value_branch[0] == "Value":
                    value = value_branch[1]
                    subtree = value_branch[2]
                    new_path = current_path + [(att_index, value)]
                    self._extract_rules(subtree, new_path, rules, attribute_names, class_name)
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
