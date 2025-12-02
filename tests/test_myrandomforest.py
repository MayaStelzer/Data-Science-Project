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
    ["Mid", "Java", "yes", "no"],
    ["Junior", "Python", "no", "yes"],
]
y = ["False", "False", "True", "True", "True", "False", "True", "False", "True", "True", "True", "True", "True", "False"]

def test_fit_selects_n_best():
    rf = MyRandomForestClassifier(n_estimators=20, max_features=2, n_best=7, random_state=42)
    rf.fit(X, y)
    assert len(rf.trees) == 7


def test_predict_length_and_labels():
    rf = MyRandomForestClassifier(n_estimators=20, max_features=2, n_best=7, random_state=42)
    rf.fit(X, y)
    preds = rf.predict(X)
    assert len(preds) == len(y)
    assert set(preds).issubset(set(y))


def test_fit_empty_raises():
    rf = MyRandomForestClassifier()
    with pytest.raises(ValueError):
        rf.fit([], [])


def test_reproducible_with_seed():
    rf1 = MyRandomForestClassifier(n_estimators=20, max_features=2, n_best=7, random_state=123)
    rf2 = MyRandomForestClassifier(n_estimators=20, max_features=2, n_best=7, random_state=123)
    rf1.fit(X, y)
    rf2.fit(X, y)
    p1 = rf1.predict(X)
    p2 = rf2.predict(X)
    assert p1 == p2
