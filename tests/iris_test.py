import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from xdecisiontree import XDecisionTreeClassifier

def apply_rule(rule, X):
    """Check which rows of X satisfy a single rule (works with DataFrame)."""
    mask = np.ones(len(X), dtype=bool)
    for f, (lb, ub) in rule['constraints'].items():
        mask &= (X[f] > lb) & (X[f] <= ub)
    return mask

def rules_predict(rules, majority_class, X):
    """Predict class using rules and majority_class. Works with pandas DataFrame."""
    y_pred = np.full(len(X), majority_class)
    for rule in rules:
        mask = apply_rule(rule, X)
        y_pred[mask] = rule['prediction']
    return y_pred

def test_rules_match_tree():
    # Load dataset
    data = load_iris()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target

    for depth in range(1,11):
        # Fit XDecisionTreeClassifier
        clf = XDecisionTreeClassifier(max_depth=depth, random_state=42)
        clf.fit(X, y)
        clf._tree_to_rule_list()  # Extract rules

        # Check predictions
        y_tree = clf.predict(X)
        y_rules = rules_predict(clf.rules, clf.majority_class, X)

        # All predictions should match
        assert np.array_equal(y_tree, y_rules), "Rule-based predictions do not match tree predictions"

test_rules_match_tree()