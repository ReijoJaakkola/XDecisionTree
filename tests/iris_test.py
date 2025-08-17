from sklearn.datasets import load_iris
from utility import test_rules_match_tree

iris = load_iris(as_frame=True)
X = iris.data
y = iris.target
test_rules_match_tree(X,y)