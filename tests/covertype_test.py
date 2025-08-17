from sklearn.datasets import fetch_covtype
from utility import test_rules_match_tree

# Load dataset
cov = fetch_covtype(as_frame=True)
X = cov.data
y = cov.target
test_rules_match_tree(X,y)