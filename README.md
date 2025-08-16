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

# Print extracted rules
print(clf)

# Example output:
# IF petal width (cm) <= 0.800 THEN 0
# ELSE IF 0.800 < petal width (cm) <= 1.750 AND petal length (cm) <= 4.950 THEN 1
# ELSE 2
```