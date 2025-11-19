"""mysimplelinearregressor.py
@author sluitel
Reused from gsprint23

Note: is used for the regressor in MySimpleLinearRegressionClassifier
"""

import numpy as np

class MySimpleLinearRegressor:
    """
    
    A simple linear regression using the least squares method.
    
    Equation:
        y = slope * x + intercept
    
    Attributes:
        slope (float): Slope (m) of the line.
        intercept (float): Intercept (b) of the line.
    
    Notes:
        Loosely based on sklearn's LinearRegression:
            https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
        Terminology: instance = sample = row and attribute = feature = column
    """
   
    def __init__(self, slope=None, intercept=None):
        
        """
        Initialize the regressor with optional slope and intercept.
        If not provided, they will be computed with fit().
        
        Args:
            slope (float): The slope (m) of the line. None if to be computed.
            intercept (float): The intercept (b) of the line. None if to be computed.
     
        """
        self.slope = slope
        self.intercept = intercept

    def fit(self, X_train, y_train):
        """
         Fits a simple linear regression line to X_train and y_train

        Args:
            X_train (list of list of numeric values): Each inner list represents a single sample with one feature.
            Shape: (n_train_samples, n_features)
            Example for 4 samples: [[1], [2], [3], [4]]

            y_train (list of numeric values): Target values corresponding to each sample in X_train.
            Shape: (n_train_samples,)
            Example: [2, 4, 6, 8]

        Notes:
        - In simple linear regression, each sample has only one feature (n_features = 1).
        - Think of X_train as a column of input values and y_train as a column of output values.
        - Using a "list of list" structure keeps the interface consistent with multi-feature datasets.

        """
        X_train = [x[0] for x in X_train] # convert 2D list with 1 col to 1D list
        self.slope, self.intercept = MySimpleLinearRegressor.compute_slope_intercept(X_train,
            y_train)

    def predict(self, X_test):
        """
        Predict values for test data.
        
        Args:
            X_test (list of list of numeric vals): Test samples. Shape = (n_test_samples, 1).
        
        Returns:
            list of numeric vals: Predicted target values.
        Raises:
            ValueError: If model is not yet fitted (slope or intercept is None)
        
        """
        if self.slope is None or self.intercept is None:
            raise ValueError("Model is not fitted yet. Call fit() before predict().")
        
        return [self.slope * x[0] + self.intercept for x in X_test]
    

    @staticmethod # decorator to denote this is a static (class-level) method
    def compute_slope_intercept(x, y):
        """
        Compute slope (m) and intercept (b) using least squares.
        This is strictly for **univariate regression** (1 predictor).

        Args:
            x (list of numeric vals): Feature values (1D list).
            y (list of numeric vals): Target values (1D list).
        
        Returns: 
            tuple:
                m (float): Slope of the line.
                b (float): Intercept of the line.

        """
        mean_x = np.mean(x)
        mean_y = np.mean(y)
        m = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(len(x))) \
            / sum((x[i] - mean_x) ** 2 for i in range(len(x)))
        # y = mx + b => y - mx
        b = mean_y - m * mean_x
        return m, b
