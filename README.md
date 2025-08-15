# XDecisionTree

`XDecisionTreeClassifier` is an extension of scikit-learn's `DecisionTreeClassifier` that extracts human-readable decision rules from trained trees. It allows you to see the model logic in a very compact **IF–ELSE** format.

## Features
- Extract rules from decision trees with readable feature names.
- Generalizes rules to remove redundant constraints.
- Prints rules in intuitive `IF ... THEN ... ELSE ...` statements.

## Installation
```bash
pip install git+https://github.com/yourusername/RuleDecisionTreeClassifier.git
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
clf.fit(X, y, feature_names=feature_names)

# Print extracted rules
print(clf)
```