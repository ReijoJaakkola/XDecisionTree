from sklearn.datasets import load_wine
from utility import test_rules_match_tree

# Load dataset
wine = load_wine(as_frame=True)
X = wine.data
y = wine.target
test_rules_match_tree(X,y)