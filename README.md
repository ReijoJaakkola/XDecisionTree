# XDecisionTree

`XDecisionTreeClassifier` is an extension of scikit-learn's `DecisionTreeClassifier` that extracts human-readable decision rules from trained trees. It allows you to see the model logic in a very compact **IF–ELSE** format.

## Installation
```bash
pip install .
```

## Example
```python
from xdecisiontree import XDecisionTreeClassifier
from sklearn.datasets import load_iris

# Load data
X, y = load_iris(return_X_y=True)
feature_names = ['sepal length', 'sepal width', 'petal length', 'petal width']

# Train model
clf = XDecisionTreeClassifier(max_depth=3)
clf.fit(X, y)

# Print extracted rules
clf.tree_to_rule_list(feature_names)
print(clf)

# Example output:
# IF petal length <= 2.450 THEN 0
# ELSE IF 2.450 < petal length <= 4.950 AND petal width <= 1.750 THEN 1
# ELSE 2
```