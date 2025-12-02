import pytest

from mysklearn.myclassifiers import MyRandomForestClassifier

header_interview = ["level", "lang", "tweets", "phd", "interviewed_well"]
X = [
    ["Senior", "Java", "no", "no"],
    ["Senior", "Java", "no", "yes"],
    ["Mid", "Python", "no", "no"],
    ["Junior", "Python", "no", "no"],
    ["Junior", "R", "yes", "no"],
    ["Junior", "R", "yes", "yes"],
    ["Mid", "R", "yes", "yes"],
    ["Senior", "Python", "no", "no"],
    ["Senior", "R", "yes", "no"],
    ["Junior", "Python", "yes", "no"],
    ["Senior", "Python", "yes", "yes"],
    ["Mid", "Python", "no", "yes"],
    ["Mid", "Java", "yes", "0"],
    ["Junior", "Python", "no", "yes"],
]
y = ["False", "False", "True", "True", "True", "False", "True", "False", "True", "True", "True", "True", "True", "False"]

def test_stratified_split():
    rf = MyRandomForestClassifier(random_state=1)
    X_train, y_train = rf.stratified_split(X, y)
    X_test, y_test = rf.test_set
    # test set size roughly 1/3
    assert abs(len(X_test) - len(X)/3) <= 1
    # each class appears in train and test
    for label in set(y):
        assert label in y_train
        assert label in y_test

def test_bootstrap_sample():
    rf = MyRandomForestClassifier(random_state=1)
    X_sample, y_sample = rf.bootstrap_sample(X, y)
    assert len(X_sample) == len(X)
    assert len(y_sample) == len(y)

def test_random_features():
    rf = MyRandomForestClassifier(F=2, random_state=1)
    features = rf.random_features(5)
    assert len(features) == 2
    assert all(f < 5 for f in features)

def test_fit_predict():
    rf = MyRandomForestClassifier(N=5, M=3, F=2, random_state=1)
    rf.fit(X, y)
    assert len(rf.trees) == 3
    X_test, y_test = rf.test_set
    y_pred = rf.predict(X_test)
    assert len(y_pred) == len(X_test)
    # predictions should be in original labels
    assert all(pred in set(y) for pred in y_pred)
