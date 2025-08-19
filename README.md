# XDecisionTree

`XDecisionTreeClassifier` is an extension of scikit-learn's `DecisionTreeClassifier` that extracts human-readable decision rules from trained trees. It allows you to see the model logic in a very compact **IF–ELSE** format.

## Installation
```bash
pip install .
```

## Example
```python
import pandas as pd
from xdecisiontree import XDecisionTreeClassifier
from sklearn import datasets

# Load data
iris = datasets.load_iris()
X = pd.DataFrame(data = iris.data, columns = iris.feature_names)
y = iris.target

# Train model
clf = XDecisionTreeClassifier(max_depth=3)
clf.fit(X, y)

# Print classifier using IF-ELSE syntax.
print(clf)

# Example output:
# IF (petal width (cm) <= 0.800) THEN 0
# ELSE IF (0.800 < petal width (cm) <= 1.750) AND (petal length (cm) <= 4.950) THEN 1
# ELSE 2

print()

# Print rules obtained by the classifier along with their support and accuracy.
clf.print_rules_with_scores(X,y)

# Example output:
# IF (petal width (cm) <= 0.800) THEN 0 (support=50, accuracy=100.00%)
# IF (0.800 < petal width (cm) <= 1.750) AND (petal length (cm) <= 4.950) THEN 1 (support=48, accuracy=97.92%)
# IF (petal width (cm) > 1.750) THEN 2 (support=46, accuracy=97.83%)
# IF (petal width (cm) > 1.750) THEN 2 (support=46, accuracy=97.83%)
# IF (0.800 < petal width (cm) <= 1.750) AND (petal length (cm) > 4.950) THEN 2 (support=6, accuracy=66.67%)
```