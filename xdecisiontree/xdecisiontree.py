import numpy as np
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
from operator import itemgetter

class XDecisionTreeClassifier(DecisionTreeClassifier):
    """
    Decision Tree Classifier extension that can extract interpretable rules.

    Attributes
    ----------
    rules : list of dict
        Extracted generalized rules after calling `tree_to_rule_list`.
        Each rule is a dict with 'constraints' and 'prediction'.
    majority_class : int
        Class predicted by default if no rule applies.
    """

    def __init__(
        self,
        criterion="gini",
        splitter="best",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=None,
        random_state=None,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        ccp_alpha=0.0,
    ):
        super().__init__(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            random_state=random_state,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            ccp_alpha=ccp_alpha,
        )
        self.rules_ = None
        self.majority_class_ = None

    def _tree_to_rule_list(self):
        """
        Extracts generalized IF-ELSE rules from the trained decision tree.
        """
        tree = self.tree_
        n_features = self.n_features_in_
        name_map = {i: self.feature_names_in_[i] for i in range(n_features)}

        def get_paths(node=0, path=None):
            if path is None:
                path = []
            if tree.feature[node] == _tree.TREE_UNDEFINED:
                value = tree.value[node][0]
                prediction = self.classes_[np.argmax(value)]
                return [(path, prediction)]
            else:
                feature = tree.feature[node]
                threshold = tree.threshold[node]
                left_paths = get_paths(tree.children_left[node], path + [(feature, -np.inf, threshold)])
                right_paths = get_paths(tree.children_right[node], path + [(feature, threshold, np.inf)])
                return left_paths + right_paths

        def is_unique_prediction(tree, constraints, prediction):
            stack = [(0, constraints.copy())]
            while stack:
                node, c = stack.pop()
                if tree.feature[node] == _tree.TREE_UNDEFINED:
                    pred = np.argmax(tree.value[node][0])
                    if pred != prediction:
                        return False
                else:
                    f = tree.feature[node]
                    th = tree.threshold[node]
                    lb, ub = c.get(f, (-np.inf, np.inf))
                    if lb < th:
                        new_c = c.copy()
                        new_c[f] = (lb, min(ub, th))
                        stack.append((tree.children_left[node], new_c))
                    if ub >= th:
                        new_c = c.copy()
                        new_c[f] = (max(lb, th), ub)
                        stack.append((tree.children_right[node], new_c))
            return True

        thresholds = {f: sorted(set(tree.threshold[tree.feature == f])) for f in range(n_features)}

        def generalize_path(path, prediction):
            constraints = {}
            for f, lb, ub in path:
                if f in constraints:
                    constraints[f][0] = max(constraints[f][0], lb)
                    constraints[f][1] = min(constraints[f][1], ub)
                else:
                    constraints[f] = [lb, ub]

            # Try removing features
            for f in list(constraints.keys()):
                test_constraints = constraints.copy()
                del test_constraints[f]
                if is_unique_prediction(tree, test_constraints, prediction):
                    del constraints[f]

            # Expand bounds
            for f in list(constraints.keys()):
                lb, ub = constraints[f]
                all_th = thresholds.get(f, [])

                def closest_index(thresholds, val, side):
                    arr = np.array(thresholds)
                    if side == 'left':
                        idx = np.searchsorted(arr, val, side='left')
                        return max(0, min(idx, len(arr)-1))
                    else:
                        idx = np.searchsorted(arr, val, side='right') - 1
                        return max(0, min(idx, len(arr)-1))

                # Expand lower bound
                if lb > -np.inf and all_th:
                    l = 0
                    r = closest_index(all_th, lb, 'left')
                    best = lb
                    while l <= r:
                        m = (l + r) // 2
                        test_lb = all_th[m]
                        test_constraints = constraints.copy()
                        test_constraints[f] = [test_lb, ub]
                        if is_unique_prediction(tree, test_constraints, prediction):
                            best = test_lb
                            r = m - 1
                        else:
                            l = m + 1
                    constraints[f][0] = best

                # Expand upper bound
                if ub < np.inf and all_th:
                    l = closest_index(all_th, ub, 'right')
                    r = len(all_th) - 1
                    best = ub
                    while l <= r:
                        m = (l + r) // 2
                        test_ub = all_th[m]
                        test_constraints = constraints.copy()
                        test_constraints[f] = [constraints[f][0], test_ub]
                        if is_unique_prediction(tree, test_constraints, prediction):
                            best = test_ub
                            l = m + 1
                        else:
                            r = m - 1
                    constraints[f][1] = best

            return constraints

        def path_satisfies(rule_constraints, path):
            for f, lb, ub in path:
                if f in rule_constraints:
                    r_lb, r_ub = rule_constraints[f]
                    if not (r_lb <= lb and ub <= r_ub):
                        return False
            return True

        rules = []
        existing_constraints = []

        paths = get_paths()
        prediction_counts = {}
        for path, prediction in paths:
            skip = False
            for rc in existing_constraints:
                if path_satisfies(rc, path):
                    skip = True
                    break
            if skip:
                continue

            if prediction not in prediction_counts:
                prediction_counts[prediction] = 0
            prediction_counts[prediction] += 1

            constraints = generalize_path(path, prediction)
            named_constraints = {name_map[f]: tuple(bounds) for f, bounds in constraints.items()}

            rules.append({'constraints': named_constraints, 'prediction': prediction})
            existing_constraints.append(constraints)

        self.rules_ = rules
        self.majority_class_ = max(prediction_counts.items(), key=lambda x: x[1])[0]

    def fit(self, X, y, sample_weight=None, check_input=True):
        super().fit(X, y, sample_weight=sample_weight, check_input=check_input)
        self._tree_to_rule_list()
        return self

    def _rules_to_str(self):
        """
        Returns a human-readable IF-ELSE rule representation of the tree.
        Uses the stored `rules` and `majority_class`.
        """
        try:
            check_is_fitted(self)
        except NotFittedError:
            raise NotFittedError("This classifier is not fitted yet. Call 'fit' before using this method.")

        first = True
        lines = []
        for _, rule in enumerate(self.rules_):
            if rule['prediction'] == self.majority_class_:
                continue
            conds = []
            for f, (lb, ub) in rule['constraints'].items():
                if lb != -np.inf and ub != np.inf:
                    conds.append(f'({lb:.3f} < {f} <= {ub:.3f})')
                elif lb != -np.inf:
                    conds.append(f'({f} > {lb:.3f})')
                elif ub != np.inf:
                    conds.append(f'({f} <= {ub:.3f})')
            prefix = 'IF' if first else 'ELSE IF'
            lines.append(f"{prefix} {' AND '.join(conds)} THEN {rule['prediction']}")
            first = False
        lines.append(f'ELSE {self.majority_class_}')
        return '\n'.join(lines)

    def __str__(self) -> str:
        return self._rules_to_str()

    def print_rules_with_scores(self, X, y):
        """
        Print all rules along with their support and accuracy, sorted by accuracy.

        Parameters
        ----------
        X : pandas.DataFrame
            Feature data.
        y : array-like
            True labels.
        """
        try:
            check_is_fitted(self)
        except NotFittedError:
            raise NotFittedError("This classifier is not fitted yet. Call 'fit' before using this method.")

        def apply_rule(rule, X):
            """Check which rows of X satisfy a single rule (works with DataFrame)."""
            mask = np.ones(len(X), dtype=bool)
            for f, (lb, ub) in rule['constraints'].items():
                mask &= (X[f] > lb) & (X[f] <= ub)
            return mask

        results = []

        for i, rule in enumerate(self.rules_):
            mask = apply_rule(rule, X)
            support = mask.sum()

            if support == 0:
                accuracy = 0.0
            else:
                accuracy = (y[mask] == rule['prediction']).mean()

            results.append({
                'index': i,
                'rule': rule,
                'support': support,
                'accuracy': accuracy
            })

        # Sort by accuracy descending
        results.sort(key=itemgetter('accuracy'), reverse=True)

        for res in results:
            i = res['index']
            rule = res['rule']
            conds = []
            for f, (lb, ub) in rule['constraints'].items():
                if lb != -np.inf and ub != np.inf:
                    conds.append(f'({lb:.3f} < {f} <= {ub:.3f})')
                elif lb != -np.inf:
                    conds.append(f'({f} > {lb:.3f})')
                elif ub != np.inf:
                    conds.append(f'({f} <= {ub:.3f})')
            condition_str = ' AND '.join(conds) if conds else 'TRUE'
            print(f"IF {condition_str} THEN {rule['prediction']} "
                f"(support={res['support']}, accuracy={res['accuracy']:.2%})")