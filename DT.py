"""
Implementation of the CART algorithm to train decision tree classifiers.
Forked from https://github.com/joachimvalente/decision-tree-cart.
"""

import numpy as np
import pandas as pd
from tqdm import tqdm

class DecisionTreeClassifier:
    def __init__(self, 
                 max_depth =                 np.inf, 
                 min_samples_split =         2,
                 min_samples_leaf =          1,
                 min_weight_fraction_split = 0.,
                 split_mode =                'greedy',
                 min_impurity_decrease =     0.,
                 pw_class_loss =             {},
                 criticality_weight =        0,
                 criticality_method =        'absolute',
                 #weight_predictions =        True,
                 ):

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split 
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_split = min_weight_fraction_split
        self.split_mode = split_mode
        self.min_impurity_decrease = min_impurity_decrease
        self.pw_class_loss = pw_class_loss
        self.criticality_weight = criticality_weight
        self.criticality_method = criticality_method
        #self.weight_predictions = weight_predictions

# ===================================================================================================================
# PREDICTION AND SCORING

    def predict(self, X, extra=None):
        """Predict class for X."""
        shp = np.shape(X)
        if len(shp)==1 or np.shape(X)[0]==1: return self._predict(X, extra)
        else: return [self._predict(inputs, extra) for inputs in X]


    def score(self, X, y, sample_weight=[]):
        """Score classification performance on a dataset."""
        X, y = np.array(X), np.array(y)
        correct = [self._predict(inputs, None) for inputs in X] == y
        if sample_weight == []: return sum(correct) / y.size
        else: return np.dot(correct, sample_weight) / np.sum(sample_weight)
             

    def _predict(self, inputs, extra):
        """Predict class for a single sample, optionally returning the decision path or explanation."""
        node = self.tree_
        if extra: path = []
        while node.left:
            if inputs[node.feature_index] < node.threshold: child = node.left; lr = 0
            else: child = node.right; lr = 1
            if extra and node.feature_index != None: path.append((node.feature_index, node.threshold, lr))
            node = child
        # Return prediction alongside some extra information.
        if extra == 'explain': return node.predicted_class, path, self._tests_to_explanation(path)
        elif extra == 'leaf_uid': return node.predicted_class, int(''.join(['1'] + [str(n[2]) for n in path]), 2) # Adding 1 at start ensures 0s don't get chopped off.
        elif extra == 'class_probs': return node.weighted_class_counts / np.sum(node.weighted_class_counts)
        # Just return prediction.
        return node.predicted_class

# ===================================================================================================================
# LOADING FROM CODE.

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
        self.num_classes = len(self.class_index)

# ===================================================================================================================
# FITTING TO A DATASET.

    def fit(self, X, y, sample_weight=[], Q=[], feature_names=None):
        """Build decision tree classifier. Require feature names as there are very useful in future."""

        # Set up basic variables.
        X = np.array(X)
        if len(X.shape) == 1: X = X.reshape(-1,1)
        self.num_samples_total, self.num_features = X.shape
        self.class_index = {c:i for i,c in enumerate(sorted(list(set(y) | set(self.pw_class_loss))))} # Alphanumeric order. Include those in pw_class_loss even if they're not in the dataset.
        self.num_classes = len(self.class_index)
        y = np.array([self.class_index[c] for c in y]) # Convert class representation into indices.
        if self.criticality_weight > 0:
            assert Q != []; Q = np.array(Q)
        else:
            assert Q == []; Q = np.zeros(X.shape[0])
        if sample_weight == []:
            sample_weight = np.ones(X.shape[0])
        else:
            sample_weight = np.array(sample_weight)
        self.total_weight = np.sum(sample_weight)
        if feature_names != None:
            self.feature_names = feature_names
        else:
            self.feature_names = np.arange(X.shape[1])

        # Set up min samples for split / leaf if these were specified as ratios.
        if type(self.min_samples_split) == float: self.min_samples_split_abs = int(np.ceil(self.min_samples_split * self.num_samples_total))
        else: self.min_samples_split_abs = self.min_samples_split
        if type(self.min_samples_leaf) == float: self.min_samples_leaf_abs = int(np.ceil(self.min_samples_leaf * self.num_samples_total))
        else: self.min_samples_leaf_abs = self.min_samples_leaf

        # Create class loss matrix.
        self.class_loss_to_matrix()

        # Initialise feature importances.
        self.feature_importances = np.zeros(self.num_features)
        self.potential_feature_importances = np.zeros(self.num_features)

        # Build tree.
        self.tree_ = self._grow_tree(X, y, sample_weight, Q)

        # Normalise feature importances.
        self.feature_importances /= sum(self.feature_importances)
        self.potential_feature_importances /= sum(self.potential_feature_importances)

    
    def class_loss_to_matrix(self):
        # If unspecified, use 1 - identity matrix.
        if self.pw_class_loss == {}: self.pwl = 1 - np.identity(self.num_classes)
        # Otherwise use values provided.
        else:
            self.pwl = np.zeros((self.num_classes,self.num_classes))
            for c,losses in self.pw_class_loss.items():
                for cc,l in losses.items():
                    # NOTE: Currently symmetric.
                    self.pwl[self.class_index[c],self.class_index[cc]] = l
                    self.pwl[self.class_index[cc],self.class_index[c]] = l
        # Normalise by max value.
        self.pwl /= np.max(self.pwl)

    
    def _grow_tree(self, X, y, sample_weight, Q, depth=0):
        """Build a decision tree by recursively finding the best split."""
        
        # Calculate a bunch of properties for this node.
        # TODO: There are definitely some repeated calculations happening here.
        N = y.size
        y_one_hot = np.zeros((N, self.num_classes))
        y_one_hot[np.arange(N), y] = 1
        class_counts = np.sum(y_one_hot, axis=0)
        weighted_class_counts = np.sum(y_one_hot*sample_weight.reshape(-1,1), axis=0)
        weight = np.sum(weighted_class_counts)
        weight_fraction = weight / self.total_weight
        
        # Get predicted class number and convert back into class label.
        # if self.weight_predictions:
        #     # If factoring in sample weights into predictions.
        predicted_class = list(self.class_index.keys())[list(self.class_index.values()).index(np.argmax(weighted_class_counts))]
        # else:
        #     # If just using class counts.
        #     predicted_class = list(self.class_index.keys())[list(self.class_index.values()).index(np.argmax(class_counts))]
        
        # Calculate impurity at this node.
        # TODO: Only need to do this at the root; can propagate down.
        imp_per_class, impurity_sum = self._impurity_sum(class_counts, weighted_class_counts)#, y, Q)
        impurity = impurity_sum / (N**2)

        # Store the node properties.
        node = Node(
            impurity=impurity,
            num_samples=N,
            class_counts=class_counts,
            weighted_class_counts = weighted_class_counts,
            weight_fraction = weight_fraction,
            predicted_class=predicted_class)
        if self.criticality_weight > 0: node.mean_Q = np.mean(Q, axis=0)

        # Split recursively until maximum depth or stopping criterion is reached.
        if impurity > 0 and depth < self.max_depth and N >= self.min_samples_split_abs and N >= 2*self.min_samples_leaf_abs and weight_fraction >= self.min_weight_fraction_split:
            
            print('Depth',depth+1)
            
            # Find the best split on each feature dimension.
            best_splits = self._best_splits(X, y, N, weight, sample_weight, Q, imp_per_class, impurity_sum)
            if best_splits is not None:   

                # Choose feature to split on.  
                impurity_decreases = [s[1] for s in best_splits]        
                if self.split_mode == 'greedy':
                    # Deterministically choose the feature with greatest impurity gain.
                    idx = np.argmax(impurity_decreases)
                elif self.split_mode == 'stochastic':
                    # Sample in proportion to impurity decrease.
                    idx = np.random.choice(range(self.num_features), p=impurity_decreases/sum(impurity_decreases))
                
                # Store the chosen feature index and threshold.
                node.feature_index = idx
                node.threshold, _, _ = best_splits[idx]

                # Split data and create two child nodes.
                indices_left = X[:, idx] < node.threshold                
                if sum(indices_left) == 0: indices_left = X[:, idx] <= node.threshold # This important line prevents creating an empty leaf due to numerical precision weirdness.
                X_left, y_left, sample_weight_left, Q_left = X[indices_left], y[indices_left], sample_weight[indices_left], Q[indices_left]
                X_right, y_right, sample_weight_right, Q_right = X[~indices_left], y[~indices_left], sample_weight[~indices_left], Q[~indices_left]                                
                node.left = self._grow_tree(X_left, y_left, sample_weight_left, Q_left, depth + 1)
                node.right = self._grow_tree(X_right, y_right, sample_weight_right, Q_right, depth + 1)

                # Store impurity decrease to measure feature importance.
                self.feature_importances[idx] += impurity_decreases[idx]
                self.potential_feature_importances += impurity_decreases

        return node


    def _best_splits(self, X, y, N, parent_weight, sample_weight, Q, imp_per_class_parent, impurity_sum_parent):
        """Find the best split location for each feature at a node."""

        # Loop through all features.
        best_splits = np.array([[None, 0, impurity_sum_parent] for idx in range(self.num_features)])       
        found_one_split = False
        for idx in range(self.num_features):

            print('   Feature',idx+1,'/',self.num_features)

            # Sort data along selected feature.
            sort = np.argsort(X[:,idx])
            thresholds = X[sort,idx]
            y_sorted = y[sort]
            sample_weight_sorted = sample_weight[sort]
            if self.criticality_weight > 0: Q_sorted = Q[indices_sorted,:]
            else: Q_sorted = Q
        
            # Initialise counts and impurities for the two children.
            # class_counts_left = np.zeros_like(class_counts_parent)
            # weighted_class_counts_left = np.zeros_like(class_counts_parent)
            # class_counts_right = class_counts_parent.copy()
            # weighted_class_counts_right = weighted_class_counts_parent.copy()
            impurity_sum_left = 0.
            impurity_sum_right = impurity_sum_parent.copy()

            imp_per_class_left = np.zeros_like(imp_per_class_parent)
            imp_per_class_right = imp_per_class_parent.copy()

            # print('START')
            # print('left = {}, impurity = {}'.format(class_counts_left, impurity_sum_left))
            # print('right = {}, impurity = {}'.format(class_counts_right, impurity_sum_right))
            # print('')
            
            # Loop through all possible split positions for this feature.
            for i in range(self.min_samples_leaf_abs, N+1-self.min_samples_leaf_abs): # Only consider splits that leave at least min_samples_leaf_abs at each child.
                c = y_sorted[i-1]
                # q = Q_sorted[i-1]
                w = sample_weight_sorted[i-1]

                w_pwl = w * self.pwl[c]
                imp_per_class_left += w_pwl
                imp_per_class_right -= w_pwl

                # Compute impurity incrementally to improve speed.
                # NOTE: Currently assumes self.pwl is symmetric. 
                impurity_sum_left += 2 * (w * imp_per_class_left[c])
                impurity_sum_right -= 2 * (w * imp_per_class_right[c])
                
                # impurity_sum_left += (w * imp_per_class_left[c]) + imp_left_in[c] 
                # impurity_sum_right -= (w * imp_per_class_right[c]) + imp_right_in[c]
                
                # Skip if this threshold is the same as the previous one.
                if thresholds[i] == thresholds[i - 1]: continue

                # The impurity of a split is the weighted average of the impurity of the children.
                # Note that conceptually, we're dividing each child twice by its population to get probabilities, then multiplying by the population again.
                # This means the visible effect is a single division.
                #impurity = ((impurity_sum_left / i) + (impurity_sum_right / (N-i))) / N 
                #weighted_impurity_decrease = (impurity_parent - impurity) * N / self.num_samples_total 

                # print('left={}, weighted={}, impurity sum={}, '.format(class_counts_left, weighted_class_counts_left, impurity_sum_left))
                # print('right={}, weighted={}, impurity sum={}, '.format(class_counts_right, weighted_class_counts_right, impurity_sum_right))
                # print('parent={}, weighted={}, impurity sum={}, '.format(class_counts_parent, weighted_class_counts_parent, impurity_sum_parent))

                # print(impurity, impurity_parent)
                # print('')

                impurity_sum = impurity_sum_left + impurity_sum_right
                #if weighted_impurity_decrease > best_splits[idx][1]:
                if impurity_sum < best_splits[idx][2]:
                    impurity_decrease = (impurity_sum_parent - impurity_sum) # * parent_weight / self.total_weight 
                    if impurity_decrease >= self.min_impurity_decrease:
                        best_splits[idx][0] = (thresholds[i] + thresholds[i - 1]) / 2 # Midpoint
                        best_splits[idx][1] = impurity_decrease
                        best_splits[idx][2] = impurity_sum 
                        if not found_one_split: found_one_split = True

        if found_one_split: return best_splits
        else: return None


    def _impurity_sum(self, class_counts, weighted_class_counts):#, y, Q):
        imp_per_class = np.inner(self.pwl, weighted_class_counts)
        return imp_per_class, np.dot(weighted_class_counts, imp_per_class) 
        #if self.criticality_weight == 0: return gini
        # return gini + (self.criticality_weight * sum(
        #                self._criticality_per_sample(class_counts, c, q)
        #                for c, q in zip(y, Q))) 


    # def _impurity_increment(self, w, class_counts, weighted_class_counts, c):#, q, other_y, other_Q): 
    #     gini = (w * np.dot(self.pwl[c], class_counts)) + np.dot(self.pwl[c], weighted_class_counts)
    #     if self.criticality_weight == 0: return gini
    #     # Need to compute criticality change both ways: from the moved point outwards, and vice versa from all the others. 
    #     # crit_from = self._criticality_per_sample(class_counts, c, q) 
    #     # crit_to = sum(self._c(qq[cc], qq[c]) for cc, qq in zip(other_y, other_Q)) 
    #     # return gini + (self.criticality_weight * (crit_from + crit_to))

        
    def _criticality_per_sample(self, class_counts, c, q):
        return sum(count * self._c(q[c], q[cc]) for cc, count in enumerate(class_counts) if cc != c and count > 0)


    def _c(self, q1, q2):
        if self.criticality_method == 'absolute':    return abs(q1 - q2)
        elif self.criticality_method == 'signed':    return q1 - q2
        elif self.criticality_method == 'rectified': return max(q1 - q2, 0)  
        else: raise Exception('Criticality method not recognised!')  

# ===================================================================================================================
# PRUNING.

    def prune_to_depth(self, depth_lim):
        '''Removes all nodes beyond a specified depth.'''
        def recurse(node, depth=0):
            if node.left != None:
                if depth == depth_lim: 
                    node.left = None; node.right = None; node.feature_index = None; node.threshold = None
                else:
                    recurse(node.left, depth+1)
                    recurse(node.right, depth+1)
        recurse(self.tree_)


    def prune_zero_cost(self):
        '''Removes sibling nodes that return the same class,
        thereby incurring zero cost in terms of training set accuracy .'''
        def recurse(node, depth=1):
            if node.left != None:
                # For decision nodes.
                left_class, left_is_leaf = recurse(node.left, depth+1)
                right_class, right_is_leaf = recurse(node.right, depth+1)
                if left_is_leaf and right_is_leaf and left_class == right_class:
                    node.left = None; node.right = None; node.feature_index = None; node.threshold = None
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
                costs.append((path_to_here, (node_weighted_impurity - leaves_weighted_impurity) / (num_leaves + 1)))
                return leaves_weighted_impurity, num_leaves
            return node_weighted_impurity, 1
        def recurse_to_prune(node, path_from_here):
            if path_from_here == []:    
                node.left = None; node.right = None; node.feature_index = None; node.threshold = None
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
            new_feature_importances[new] = self.feature_importances[old]
            new_potential_feature_importances[new] = self.potential_feature_importances[old]
        self.feature_names = new_feature_names
        self.feature_importances = new_feature_importances
        self.potential_feature_importances = new_potential_feature_importances
        recurse_overwrite(self.tree_, reindex)

# ===================================================================================================================
# TOOLS FOR EXPLORATION AND VISUALISATION.

    def survey(self):
        """Extract basic information from the tree: number of usages of each feature and at what depth."""
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

    
    def get_node_info(self, path=None, uid=None):
        """Given a path (binary string representing left/right decisions) or UID (integer encoding of path),
        return the path and factual explanation for that node, 
        as well as the feature and threshold tested (if decision node) and the predicted class."""
        if not path: path = str(bin(uid))[3:] # Chop off 0b as well as leading 1.
        node = self.tree_; tests = []
        for lr in path: 
            lr = int(lr)
            if lr == 0: child = node.left; 
            else: child = node.right
            tests.append((node.feature_index, node.threshold, lr))
            node = child
        return tests, self._tests_to_explanation(tests), node


    def _tests_to_explanation(self, path):
        explanans = {}
        for i, t, lr in path:
            f = self.feature_names[i]
            if f not in explanans: explanans[f] = [-np.inf,np.inf]
            explanans[f][1-lr] = t
        return explanans


    def to_ascii(self, better_feature_names=None, class_values=None, show_details=True, out_file=None):
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


    def to_code(self, comment=False, class_values=None, out_file=None): 
        """Print tree rules as an executable function definition.
        Adapted from https://stackoverflow.com/questions/20224526/how-to-extract-the-decision-rules-from-scikit-learn-decision-tree"""

        lines = []
        #lines.append("def tree({}):".format('ARGS'))#", ".join([f for f in self.feature_names])))

        def recurse(node, depth=0):
            indent = "    " * depth
            pred = node.predicted_class
            if comment:
                class_counts = node.class_counts
                conf = int(100 * (class_counts[self.class_index[pred]] / np.sum(class_counts)))
                weighted_impurity = node.impurity * node.num_samples / self.num_samples_total
            if class_values != None: pred = class_values[pred]
            # Decision nodes.
            if node.feature_index != None:
                feature = self.feature_names[node.feature_index]
                threshold = node.threshold
                if comment:
                    lines.append("{}if {} < {}: # depth = {}, best class = {}, confidence = {}%, class counts = {}, weighted impurity = {}".format(indent, feature, threshold, depth, pred, conf, class_counts.astype(int), weighted_impurity))
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
                    lines.append("{}return {} # confidence = {}%, counts = {}, weighted impurity = {}".format(indent, pred, conf, class_counts.astype(int), weighted_impurity))
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
                class_counts = node.class_counts
                pred = node.predicted_class
                prob =  num_samples / self.num_samples_total
                conf = max(class_counts) / num_samples
                weighted_impurity = node.impurity * prob
                row += [num_samples, class_counts, pred, prob, conf, node.impurity, weighted_impurity]
                data.append(row)
        recurse(self.tree_)
        df = pd.DataFrame(data,columns=['uid','depth'] + [f+sign for f in self.feature_names for sign in [' >',' <']] + ['num_samples','class counts','class','prob','conf','impurity','weighted impurity']).set_index('uid')
        
        # If no out file specified, just return.
        if out_file == None: return df
        else: df.to_csv(out_file+'.csv', index=False)

    
    def to_rectangles(self, feature_names, lims, ax=None, visualise=True, class_colours=None):
        """Return box specifications for visualising the decision boundary across two features."""
        assert len(feature_names) == len(lims) == 2
        if visualise:
            assert class_colours != None # Need colour spec.
            import matplotlib.pyplot as plt
            from matplotlib.patches import Rectangle
            if not ax: _, ax = plt.subplots()
            ax.set_xlim(lims[0]); ax.set_ylim(lims[1])
            ax.set_xlabel(feature_names[0]); ax.set_ylabel(feature_names[1])
        boxes = {}
        for uid, leaf in self.to_dataframe().iterrows():
            xy = []
            for i, (f, lim) in enumerate(zip(feature_names, lims)):
                min_val = max(leaf['{} >'.format(f)], lim[0])
                max_val = min(leaf['{} <'.format(f)], lim[1])
                xy.append(min_val)
                if i == 0: width = max_val - min_val 
                else:      height = max_val - min_val 
            boxes[uid] = {'xy':xy, 'width':width, 'height':height, 'class': leaf['class']}
            if visualise:
                ax.add_patch(Rectangle(xy=xy, 
                             width=width, 
                             height=height, 
                             facecolor=class_colours[leaf['class']],
                             edgecolor='k',
                             zorder=-10))
        return boxes


# ===================================================================================================================
# NODE CLASS.


class Node:
    def __init__(self, impurity, num_samples, class_counts, weighted_class_counts, weight_fraction, predicted_class):
        self.impurity = impurity
        self.num_samples = num_samples
        self.class_counts = class_counts
        self.weighted_class_counts = weighted_class_counts
        self.weight_fraction = weight_fraction
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
                str(self.class_counts),
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