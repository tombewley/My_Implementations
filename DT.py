"""
Implementation of the CART algorithm to train decision tree classifiers.
Forked from https://github.com/joachimvalente/decision-tree-cart.
"""

import numpy as np
import pandas as pd


class DecisionTreeClassifier:
    def __init__(self, 
                 max_depth =                np.inf, 
                 min_samples_split =        2,
                 min_samples_leaf =         1,
                 impurity_metric =          'gini',
                 split_mode =               'greedy',
                 min_impurity_decrease =    0.,
                 pw_class_loss =            None
                 ):

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split 
        self.min_samples_leaf = min_samples_leaf
        self.impurity_metric = impurity_metric
        self.split_mode = split_mode
        self.min_impurity_decrease = min_impurity_decrease
        self.pw_class_loss = pw_class_loss


    def class_loss_to_matrix(self):
        # If unspecified, use 1 - Identity matrix.
        if self.pw_class_loss == None: self._pwl = 1 - np.identity(self.n_classes)
        # Otherwise use values provided.
        else:
            self._pwl = np.zeros((self.n_classes,self.n_classes))
            for c,losses in self.pw_class_loss.items():
                for cc,l in losses.items():
                    # *** Must be symmetric! ***
                    self._pwl[self.class_index[c],self.class_index[cc]] = l
                    self._pwl[self.class_index[cc],self.class_index[c]] = l
        # Normalise by max value.
        self._pwl /= np.max(self._pwl)


    def fit(self, X, y, feature_names):
        """Build decision tree classifier. Require feature and class names as there are very useful in future."""

        # Set up basic variables.
        self.num_samples_total, self.num_features = X.shape
        self.feature_names = feature_names
        self.class_index = {c:i for i,c in enumerate(sorted(list(set(y))))} # Alphanumeric order.
        self.n_classes = len(self.class_index)
        X, y = np.array(X), np.array([self.class_index[c] for c in y]) # Convert class representation into indices.

        # Set up min samples for split / leaf if these were specified as ratios.
        if type(self.min_samples_split) == float: self.min_samples_split_abs = int(np.ceil(self.min_samples_split * self.num_samples_total))
        else: self.min_samples_split_abs = self.min_samples_split
        if type(self.min_samples_leaf) == float: self.min_samples_leaf_abs = int(np.ceil(self.min_samples_leaf * self.num_samples_total))
        else: self.min_samples_leaf_abs = self.min_samples_leaf

        # Create class loss matrix.
        self.class_loss_to_matrix()

        # Initialise feature importances.
        self.feature_importances_ = np.zeros(self.num_features)
        self.potential_feature_importances_ = np.zeros(self.num_features)

        # Build tree.
        self.tree_ = self._grow_tree(X, y)

        # Normalise feature importances.
        self.feature_importances_ /= sum(self.feature_importances_)
        self.potential_feature_importances_ /= sum(self.potential_feature_importances_)


    def predict(self, X, extra=None):
        """Predict class for X."""
        shp = np.shape(X)
        if len(shp)==1 or np.shape(X)[1]==1: return self._predict(X, extra)
        else: return [self._predict(inputs, extra) for inputs in X]


    def score(self, X, y):
        """Score classification performance on a dataset."""
        X, y = np.array(X), np.array(y)
        correct = [self._predict(inputs, None) for inputs in X] == y
        return sum(correct) / y.size


    def leaf_path(self, leaf_uid):
        """Given a leaf UID (integer encoding of binary string representing leaf/right decisions)
        Return the path and factual explanation for that leaf."""
        node = self.tree_; path = []
        for lr in str(bin(leaf_uid))[3:]: # Chop off 0b as well as leading 1.
            lr = int(lr)
            if lr == 0: child = node.left; 
            else: child = node.right
            path.append((node.feature_index, node.threshold, lr))
            node = child
        return path, self._path_to_explanation(path)


    def debug(self, better_feature_names=None, class_values=None, show_details=True, out_file=None):
        """Create ASCII visualization of decision tree."""
        if better_feature_names:    feature_names = [better_feature_names[f] for f in self.feature_names]
        else:                       feature_names = self.feature_names
        lines = self.tree_.debug(feature_names, class_values, show_details)
        # If no out file specified, just print.
        if out_file == None: 
           for line in lines: print(line)
        else: 
            with open(out_file+'.txt', 'w', encoding='utf-8') as f:
                for l in lines: f.write(l+'\n')


    def from_code(self, lines, feature_names):
        """Use a text file containing Python-style if-then rules to define the tree structure."""
        
        self.feature_names = feature_names
        self.num_features = len(feature_names)
        self.num_samples_total = 1
        classes = set()

        def recurse(lines):
            node = Node(0, 1, [1], None)
            line = lines.pop(0)
            if line[0] == 'if':
                assert line[1] in feature_names 
                assert line[2] == '<'
                assert ':' in line[3]
                node.feature_index = feature_names.index(line[1])
                node.threshold = float(line[3][:line[3].find(':')])
                node.left, lines = recurse(lines)
                assert lines[0] == ['else:']
                lines.pop(0)
                node.right, lines = recurse(lines)
            else: 
                assert line[0] == 'return'
                try: node.predicted_class = int(line[1]) # Make predicted class an int if possible.
                except: node.predicted_class = line[1]
                classes.add(node.predicted_class)
            return node, lines

        all_lines = [[w for w in l.strip().split(' ') if w != ''] for l in lines if l.strip() != '']
        self.tree_, _ = recurse(all_lines)
        self.class_index = {c:i for i,c in enumerate(sorted(classes))} # Alphanumeric order.
        self.n_classes = len(self.class_index)


    def to_code(self, comment=False, class_values=None, out_file=None): 
        """Print tree rules as an executable function definition.
        Adapted from https://stackoverflow.com/questions/20224526/how-to-extract-the-decision-rules-from-scikit-learn-decision-tree"""

        lines = []
        #lines.append("def tree({}):".format('ARGS'))#", ".join([f for f in self.feature_names])))

        def recurse(node, depth=0):
            indent = "    " * depth
            pred = node.predicted_class
            if comment:
                counts = node.num_samples_per_class
                conf = int(100 * (counts[pred] / np.sum(counts)))
                weighted_impurity = node.impurity * node.num_samples / self.num_samples_total
            if class_values != None: pred = class_values[pred]
            # Decision nodes.
            if node.feature_index != None:
                feature = self.feature_names[node.feature_index]
                threshold = node.threshold
                if comment:
                    lines.append("{}# depth = {}, best class = {}, confidence = {}%, counts = {}, weighted impurity = {}".format(indent, depth, pred, conf, counts.astype(int), weighted_impurity))
                else:
                    lines.append("{}if {} < {}:".format(indent, feature, threshold))
                recurse(node.left, depth+1)
                if comment:
                    lines.append("{}else: # if {} >= {}".format(indent, feature, threshold))
                else:
                    lines.append("{}else:".format(indent))
                recurse(node.right, depth+1)
            # Leaf nodes.
            else:
                if comment: 
                    lines.append("{}return {} # confidence = {}%, counts = {}, weighted impurity = {}".format(indent, pred, conf, counts.astype(int), weighted_impurity))
                else: 
                    lines.append("{}return {}".format(indent, pred))

        recurse(self.tree_)

        # If no out file specified, just print.
        if out_file == None: 
           for line in lines: print(line)
        else: 
            with open(out_file+'.py', 'w', encoding='utf-8') as f:
                for l in lines: f.write(l+'\n')


    def to_dataframe(self, out_file=None):
        """Represent the leaf notes of the tree as rows of a Pandas dataframe."""
        data = []
        def recurse(node, depth=0, path=[], partitions=[]):
            # Decision nodes.
            if node.feature_index != None:
                recurse(node.left, depth+1, path+[0], partitions+[(node.feature_index, '<', node.threshold)])
                recurse(node.right, depth+1, path+[1], partitions+[(node.feature_index, '>', node.threshold)])
            # Leaf nodes.
            else:
                leaf_uid = int(''.join(['1'] + [str(n) for n in path]), 2) # <--- IMPORTANT TO ADD AN EXTRA 1 AT THE START SO THAT 0s DON'T GET CHOPPED OFF.
                row = [leaf_uid, depth]
                for f in range(self.num_features):
                    for sign in ['>','<']:
                        p_rel = [p for p in partitions if p[0] == f and p[1] == sign]
                        # The last partition for each (feature name, sign) pair is always the most restrictive.
                        if len(p_rel) > 0: val = p_rel[-1][2]
                        else: val = (np.inf if sign == '<' else -np.inf)  
                        row.append(val)
                num_samples = node.num_samples
                counts = node.num_samples_per_class
                pred = node.predicted_class
                prob =  num_samples / self.num_samples_total
                conf = max(counts) / num_samples
                weighted_impurity = node.impurity * prob
                row += [num_samples, counts, pred, prob, conf, node.impurity, weighted_impurity]
                data.append(row)
        recurse(self.tree_)
        df = pd.DataFrame(data,columns=['uid','depth'] + [f+sign for f in self.feature_names for sign in [' >',' <']] + ['num_samples','counts','class','prob','conf','impurity','weighted impurity']).set_index('uid')
        
        # If no out file specified, just return.
        if out_file == None: return df
        else: df.to_csv(out_file+'.csv', index=False)


    def survey(self):
        '''Extract basic information from the tree: number of usages of each feature and at what depth.'''
        def recurse(node, depth=1):
            nodes.append([depth,node.predicted_class])
            if node.left != None:
                nodes[-1].append(node.feature_index)
                recurse(node.left, depth+1)
                recurse(node.right, depth+1)
        nodes = []
        recurse(self.tree_)
        return nodes


    def num_leaves(self):
        def recurse(node):
            if node.left == None: return 1
            else: return recurse(node.left) + recurse(node.right)
        return recurse(self.tree_)


    def prune_zero_cost(self):
        '''Removes sibling nodes that return the same class,
        thereby incurring zero cost in terms of training set accuracy .'''
        def recurse(node, depth=1):
            if node.left != None:
                # For decision nodes.
                left_class, left_is_leaf = recurse(node.left, depth+1)
                right_class, right_is_leaf = recurse(node.right, depth+1)
                if left_is_leaf and right_is_leaf and left_class == right_class:
                    node.left = None; node.right = None; node.feature_index = None
                    return node.predicted_class, True
                return node.predicted_class, False
            return node.predicted_class, True
        recurse(self.tree_)


    def prune_min_cost_comp(self):
        '''Performs one iteration of pruning for the minimal cost complexity approach.
        See http://mlwiki.org/index.php/Cost-Complexity_Pruning for details.'''
        def recurse_compute_cost(node, path_to_here):
            node_weighted_impurity = node.impurity * node.num_samples / self.num_samples_total
            if node.left != None:
                left_leaves_weighted_impurity, left_num_leaves = recurse_compute_cost(node.left, path_to_here+[0]) 
                right_leaves_weighted_impurity, right_num_leaves = recurse_compute_cost(node.right, path_to_here+[1])
                leaves_weighted_impurity = left_leaves_weighted_impurity + right_leaves_weighted_impurity
                num_leaves = left_num_leaves + right_num_leaves
                costs.append((path_to_here, (node_weighted_impurity - leaves_weighted_impurity) / (num_leaves + 1)))#, num_leaves))
                return leaves_weighted_impurity, num_leaves
            return node_weighted_impurity, 1
        def recurse_to_prune(node, path_from_here):
            if path_from_here == []:    
                #print(self.feature_names[node.feature_index], node.threshold); 
                node.left = None; node.right = None; node.feature_index = None 
            elif path_from_here.pop(0): recurse_to_prune(node.right, path_from_here)
            else:                       recurse_to_prune(node.left, path_from_here)

        costs = []
        recurse_compute_cost(self.tree_, [])
        path_to_prune = sorted(costs, key=lambda x: x[1])[0][0]
        recurse_to_prune(self.tree_, path_to_prune)


    def reindex(self):
        '''Reindexes feature numbers so a more compact input vector can be used.'''
        def recurse_collect(node):
            if node.left != None:
                indices.add(node.feature_index)
                recurse_collect(node.left)
                recurse_collect(node.right)
        def recurse_overwrite(node, reindex):
            if node.left != None:
                node.feature_index = reindex[node.feature_index]
                recurse_overwrite(node.left, reindex)
                recurse_overwrite(node.right, reindex)
        indices = set()
        recurse_collect(self.tree_)
        reindex = {old:new for new,old in enumerate(list(indices))}
        self.num_features = len(reindex)
        new_feature_names = [None] * len(reindex)
        new_feature_importances = [None] * len(reindex)
        new_potential_feature_importances = [None] * len(reindex)
        for old,new in reindex.items(): 
            new_feature_names[new] = self.feature_names[old]
            new_feature_importances[new] = self.feature_importances_[old]
            new_potential_feature_importances[new] = self.potential_feature_importances_[old]
        self.feature_names = new_feature_names
        self.feature_importances_ = new_feature_importances
        self.potential_feature_importances_ = new_potential_feature_importances
        recurse_overwrite(self.tree_, reindex)

        
    def _impurity_sum(self, counts):
        if self.impurity_metric == 'gini': return np.dot(np.dot(counts, np.tril(self._pwl)), counts)
        raise ValueError('Invalid impurity metric!')


    def _impurity_increment(self, c, counts): 
        if self.impurity_metric == 'gini': return np.dot(self._pwl[c], counts)


    def _best_splits(self, X, y, N, num_parent, impurity_parent):
        """Find the best split location for each feature at a node."""

        # Loop through all features.
        best_splits = np.array([[None, self.min_impurity_decrease] for idx in range(self.num_features)])
        found_one_split = False
        for idx in range(self.num_features):

            print('   Feature',idx+1,'/',self.num_features)

            # Sort data along selected feature.
            thresholds, classes = zip(*sorted(zip(X[:, idx], y)))

            # Loop through all possible split positions for this feature.
            num_left = np.array([0 for c in range(self.n_classes)],dtype=np.int64)
            impurity_sum_left = 0.
            num_right = num_parent.copy()
            impurity_sum_right = self._impurity_sum(num_right)
            for i in range(self.min_samples_leaf_abs, N+1-self.min_samples_leaf_abs): # Only consider splits that leave at least min_samples_leaf_abs at each child.
                c = classes[i - 1]
                num_left[c] += 1
                num_right[c] -= 1

                # Compute impurity incrementally to improve speed.
                impurity_sum_left += self._impurity_increment(c, num_left)
                impurity_sum_right -= self._impurity_increment(c, num_right)

                # Skip if this threshold is the same as the previous one.
                if thresholds[i] == thresholds[i - 1]: continue

                # The impurity of a split is the weighted average of the impurity of the children.
                impurity = ((impurity_sum_left / i) + (impurity_sum_right / (N-i))) / N

                weighted_impurity_decrease = (impurity_parent - impurity) * N / self.num_samples_total 

                if weighted_impurity_decrease > best_splits[idx][1]:
                    best_splits[idx][0] = (thresholds[i] + thresholds[i - 1]) / 2  # midpoint
                    best_splits[idx][1] = weighted_impurity_decrease
                    if not found_one_split: found_one_split = True

        if found_one_split: return best_splits
        else: return None


    def _grow_tree(self, X, y, depth=0):
        """Build a decision tree by recursively finding the best split."""
        # Population for each class in current node. The predicted class is the one with largest population.
        N = y.size
        num_samples_per_class = np.array([np.sum(y == c) for c in range(self.n_classes)],dtype=np.int64)
        impurity = self._impurity_sum(num_samples_per_class) / (N**2)
        #impurity = self._impurity(num_samples_per_class, N)
        # Get modal class number and convert back into class label.
        predicted_class = list(self.class_index.keys())[list(self.class_index.values()).index(np.argmax(num_samples_per_class))]
        node = Node(
            impurity=impurity,
            num_samples=N,
            num_samples_per_class=num_samples_per_class,
            predicted_class=predicted_class,
        )

        # Split recursively until maximum depth or stopping criterion is reached.
        if depth < self.max_depth and N >= self.min_samples_split_abs and N >= 2*self.min_samples_leaf_abs:
            print('Depth',depth+1)
            best_splits = self._best_splits(X, y, N, num_samples_per_class, impurity)
            if best_splits is not None:    

                # Choose feature to split on.  
                impurity_decreases = [s[1] for s in best_splits]        
                if self.split_mode == 'greedy':
                    # Deterministically choose the feature with greatest impurity gain.
                    idx = np.argmax(impurity_decreases)
                elif self.split_mode == 'stochastic':
                    # Sample in proportion to impurity decrease.
                    idx = np.random.choice(range(self.num_features), p=impurity_decreases/sum(impurity_decreases))
                thr, _ = best_splits[idx]

                # Create two child nodes.
                indices_left = X[:, idx] < thr                
                if sum(indices_left) == 0: indices_left = X[:, idx] <= thr # This important line prevents creating an empty leaf due to numerical precision weirdness.
                X_left, y_left = X[indices_left], y[indices_left]
                X_right, y_right = X[~indices_left], y[~indices_left]
                node.feature_index = idx
                node.threshold = thr
                node.left = self._grow_tree(X_left, y_left, depth + 1)
                node.right = self._grow_tree(X_right, y_right, depth + 1)

                # Store impurity decrease to measure feature importance.
                self.feature_importances_[idx] += impurity_decreases[idx] 
                self.potential_feature_importances_ += impurity_decreases

        return node


    def _predict(self, inputs, extra):
        """Predict class for a single sample, optionally returning the decision path or explanation."""
        node = self.tree_
        if extra: path = []
        while node.left:
            if inputs[node.feature_index] < node.threshold: child = node.left; lr = 0
            else: child = node.right; lr = 1
            if extra and node.feature_index != None: path.append((node.feature_index, node.threshold, lr))
            node = child
        # Return prediction alongside decision path and factual explanation.
        if extra == 'explain': return node.predicted_class, path, self._path_to_explanation(path)
        elif extra == 'leaf_uid': return node.predicted_class, int(''.join(['1'] + [str(n[2]) for n in path]), 2) # <--- IMPORTANT TO ADD AN EXTRA 1 AT THE START SO THAT 0s DON'T GET CHOPPED OFF.
        # Just return prediction.
        return node.predicted_class


    def _path_to_explanation(self, path):
        explanans = {}
        for i, t, lr in path:
            f = self.feature_names[i]
            if f not in explanans: explanans[f] = [-np.inf,np.inf]
            explanans[f][1-lr] = t
        return explanans


# --------------------


class Node:
    """A decision tree node."""

    def __init__(self, impurity, num_samples, num_samples_per_class, predicted_class):
        self.impurity = impurity
        self.num_samples = num_samples
        self.num_samples_per_class = num_samples_per_class
        self.predicted_class = predicted_class
        self.feature_index = None
        self.threshold = None
        self.left = None
        self.right = None

    def debug(self, feature_names, class_values, show_details):
        """Print an ASCII visualization of the tree."""
        lines, _, _, _ = self._debug_aux(
            feature_names, class_values, show_details, root=True
        )
        return lines

    def _debug_aux(self, feature_names, class_values, show_details, root=False):
        # See https://stackoverflow.com/a/54074933/1143396 for similar code.
        is_leaf = not self.right
        if is_leaf:
            if class_values: lines = [class_values[self.predicted_class]]
            else: lines = [str(self.predicted_class)]
        else:
            lines = [
                "{},{:.2f}".format(feature_names[self.feature_index], self.threshold)
            ]
        if show_details:
            lines += [
                "impurity = {:.5f}".format(self.impurity),
                "samples = {}".format(self.num_samples),
                str(self.num_samples_per_class),
            ]
        width = max(len(line) for line in lines)
        height = len(lines)
        if is_leaf:
            lines = ["║ {:^{width}} ║".format(line, width=width) for line in lines]
            lines.insert(0, "╔" + "═" * (width + 2) + "╗")
            lines.append("╚" + "═" * (width + 2) + "╝")
        else:
            lines = ["│ {:^{width}} │".format(line, width=width) for line in lines]
            lines.insert(0, "┌" + "─" * (width + 2) + "┐")
            lines.append("└" + "─" * (width + 2) + "┘")
            lines[-2] = "┤" + lines[-2][1:-1] + "├"
        width += 4  # for padding

        if is_leaf:
            middle = width // 2
            lines[0] = lines[0][:middle] + "╧" + lines[0][middle + 1 :]
            return lines, width, height, middle

        # If not a leaf, must have two children.
        left, n, p, x = self.left._debug_aux(feature_names, class_values, show_details)
        right, m, q, y = self.right._debug_aux(feature_names, class_values, show_details)
        top_lines = [n * " " + line + m * " " for line in lines[:-2]]
        # fmt: off
        middle_line = x * " " + "┌" + (n - x - 1) * "─" + lines[-2] + y * "─" + "┐" + (m - y - 1) * " "
        bottom_line = x * " " + "│" + (n - x - 1) * " " + lines[-1] + y * " " + "│" + (m - y - 1) * " "
        # fmt: on
        if p < q:
            left += [n * " "] * (q - p)
        elif q < p:
            right += [m * " "] * (p - q)
        zipped_lines = zip(left, right)
        lines = (
            top_lines
            + [middle_line, bottom_line]
            + [a + width * " " + b for a, b in zipped_lines]
        )
        middle = n + width // 2
        if not root:
            lines[0] = lines[0][:middle] + "┴" + lines[0][middle + 1 :]
        return lines, n + m + width, max(p, q) + 2 + len(top_lines), middle