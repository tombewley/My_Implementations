"""A variant of the CART tree induction algorithm, 
augmented with features that are particular to the task of RL policy distillation:
- x
- x
- x
"""

import numpy as np
import math
import pandas as pd
pd.options.mode.chained_assignment = None
from collections import Counter
from tqdm import tqdm

class AugDT:

# ===================================================================================================================
# INITIALISATION METHODS.


    def __init__(self,
                 classifier,                                # Whether actions are discrete or continuous.
                 max_depth =                 np.inf,        # Depth at which to stop splitting.  
                 max_num_leaves =            np.inf,        # Max number of leaves in tree.
                 min_samples_split =         2,             # Min samples at a node to consider splitting. 
                 min_weight_fraction_split = 0.,            # Min weight fraction at a node to consider splitting.
                 min_samples_leaf =          1,             # Min samples at a leaf to accept split.
                 min_impurity_gain =         0.,            # Min impurity gain to accept split.
                 stochastic_splits =         False,         # Whether to samples splits proportional to impurity gain. Otherwise deterministic argmax.
                 pairwise_action_loss =      {},            # (For classification) badness of each pairwise action error. If unspecified, assume uniform.
                 gamma =                     1              # Discount factor to use when considering reward.
                 ):   
        # TODO: Some of these probably belong in the growth algorithm arguments.
        self.classifier = classifier
        self.max_depth = max_depth
        self.max_num_leaves = max_num_leaves
        self.min_samples_split = min_samples_split 
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_split = min_weight_fraction_split
        self.stochastic_splits = stochastic_splits
        self.min_impurity_gain = min_impurity_gain
        self.pairwise_action_loss = pairwise_action_loss
        self.gamma = gamma
        self.have_df = False


    def load_data(self, o, a, r=[], p=[], n=[], w=[], feature_names=None, append=False):
        """
        o = observations.
        a = actions.
        r = rewards.
        p = index of preceding sample (negative = start of trajectory).
        n = index of successor sample (negative = end of trajectory).
        w = sample weight.
        """
        if append: raise Exception('Not yet implemented.')

        # Store basic values.
        self.o, self.r, self.p, self.n = o, r, p, n
        self.num_samples, self.num_features = self.o.shape
        if w == []: self.w = np.ones(self.num_samples)
        else: self.w = w
        self.w_sum = np.sum(self.w)
        if feature_names: self.feature_names = feature_names
        else:             self.feature_names = np.arange(self.num_features).astype(str) # Initialise to indices.

        if self.classifier:
            # Define a canonical (alphanumeric) order for the actions so can work in terms of indices.
            self.action_order = sorted(list(set(a) | set(self.pairwise_action_loss)))
            self.num_actions = len(self.action_order)
            self.action_loss_to_matrix() 
            self.a = np.array([self.action_order.index(c) for c in a]) # Convert array into indices.
        else:
            self.a = a

        # Set up min samples for split / leaf if these were specified as ratios.
        if type(self.min_samples_split) == float: self.min_samples_split_abs = int(np.ceil(self.min_samples_split * self.num_samples))
        else: self.min_samples_split_abs = self.min_samples_split
        if type(self.min_samples_leaf) == float: self.min_samples_leaf_abs = int(np.ceil(self.min_samples_leaf * self.num_samples))
        else: self.min_samples_leaf_abs = self.min_samples_leaf

        # Initialise feature importances.
        #self.feature_importance = np.zeros(self.num_features)
        #self.potential_feature_importance = np.zeros(self.num_features)

        # Compute return for each sample in the dataset.
        self.compute_returns()


    def action_loss_to_matrix(self):
        """Convert dictionary representation to a matrix for use in action_impurity calculations."""
        # If unspecified, use 1 - identity matrix.
        if self.pairwise_action_loss == {}: self.pwl = 1 - np.identity(self.num_actions)
        # Otherwise use values provided.
        else:
            self.pwl = np.zeros((self.num_actions,self.num_actions))
            for c,losses in self.pairwise_action_loss.items():
                ci = self.action_order.index(c)
                for cc,l in losses.items():
                    cci = self.action_order.index(cc)
                    # NOTE: Currently symmetric.
                    self.pwl[ci,cci] = l
                    self.pwl[cci,ci] = l


# ===================================================================================================================
# COMPLETE GROWTH ALGORITHMS.


    def grow_depth_first(self, o, a, r=[], p=[], n=[], w=[], by='action', feature_names=None):
        """
        Accept a complete dataset and grow a complete tree depth-first as in CART.
        """
        assert by in ('action','value','best')
        if by in ('value','best'): assert r != [], 'Need reward information to split by value.'
        self.load_data(o, a, r, p, n, w, feature_names, append=False)
        self.seed()
        def recurse(node, depth):
            if depth < self.max_depth and self.num_leaves < self.max_num_leaves and self.split(node, by):
                    self.num_leaves += 1
                    recurse(node.left, depth+1)
                    recurse(node.right, depth+1)
        recurse(self.tree, 0)

    
    #def grow_best_first()

    '''
    TODO: Other algorithms: 
        - Best first (i.e. highest impurity), one step at a time.
            - Could also define 'best' as highest impurity gain, but this requires us to try splitting every node first!
        - Best first with active sampling (a la TREPAN).
        - 
    '''


# ===================================================================================================================
# GENERIC METHODS FOR GROWTH.


    def seed(self):
        """Initialise a new tree with its root node."""
        assert hasattr(self, 'o'), 'No data loaded.' 
        self.tree = self.new_leaf([], np.arange(self.num_samples))
        self.num_leaves = 1

    
    def new_leaf(self, address, indices):
        """Create a new leaf, computing attributes where required."""
        node = Node(address = address,
                    indices = indices,
                    num_samples = len(indices),
                    )
        # Action attributes for classification.
        if self.classifier: 
            # Action counts, unweighted and weighted.
            a_one_hot = np.zeros((len(indices), self.num_actions))
            a_one_hot[np.arange(len(indices)), self.a[indices]] = 1
            node.action_counts = np.sum(a_one_hot, axis=0)
            node.weighted_action_counts = np.sum(a_one_hot*self.w[indices].reshape(-1,1), axis=0)
            # Modal action from argmax of weighted_action_counts.
            node.action_best = np.argmax(node.weighted_action_counts)
            # Action probabilities from normalising weighted_action_counts.
            node.action_probs = node.weighted_action_counts / np.sum(node.weighted_action_counts)
            # Action impurity attributes.
            node.per_sample_action_loss = np.inner(self.pwl, node.weighted_action_counts) # This is the increase in action_impurity that would result from adding one sample of each action class.
            node.action_impurity_sum = np.dot(node.weighted_action_counts, node.per_sample_action_loss)  
            node.action_impurity = node.action_impurity_sum / (node.num_samples**2)
        # Action attributes for regression.
        else:
            node.action_best = np.mean(self.a[indices])
            var = np.var(self.a[indices])
            node.action_impurity_sum = var * node.num_samples
            node.action_impurity = math.sqrt(var) # NOTE: Using standard deviation!
        # Value attributes.
        if self.g != []:
            node.value_mean = np.mean(self.g[indices])
            var = np.var(self.g[indices])
            node.value_impurity_sum = var * node.num_samples
            node.value_impurity = math.sqrt(var) # NOTE: Using standard deviation!
        else: node.value_mean = 0; node.value_impurity = 0
        return node


    def split(self, node, by):
        """Split a leaf node."""
        assert node.left == None, 'Not a leaf node.'
        # Iterate through features and find best split for each.
        candidate_splits = []
        for f in range(self.num_features):
            # When doing 'best', we try both 'action' and 'value',
            # and choose whichever gives the highest impurity gain as a *ratio* of the parent impurity.
            if by in ('action','best'):
                candidate_splits.append(self.find_best_split_per_feature(node, f, 'action'))
                if candidate_splits[-1][3][0] != None: # To prevent divide by 0.
                    candidate_splits[-1][3][1] /= node.action_impurity # Relative.
            if by in ('value','best'):
                candidate_splits.append(self.find_best_split_per_feature(node, f, 'value'))
                if candidate_splits[-1][3][0] != None: # To prevent divide by 0.
                    candidate_splits[-1][3][1] /= node.value_impurity # Relative.
        if sum([s[3][0] != None for s in candidate_splits]) > 0: # If beneficial split found on at least one feature.
            # Choose one feature to split on.  
            relative_impurity_gains = [s[3][1] for s in candidate_splits]  
            if self.stochastic_splits:
                # Sample in proportion to impurity gain.
                chosen_split = np.random.choice(range(len(candidate_splits)), p=relative_impurity_gains/sum(relative_impurity_gains))
            else:
                # Deterministically choose the feature with greatest imourity gain.
                chosen_split = np.argmax(relative_impurity_gains) # Ties broken by lowest index.                
            # Unpack information for this split and create child leaves.
            node.feature_index, node.split_by, indices_sorted, (node.threshold, _, split_index) = candidate_splits[chosen_split]            
            node.left = self.new_leaf(node.address+[0], indices_sorted[:split_index])
            node.right = self.new_leaf(node.address+[1], indices_sorted[split_index:])        
            # Store action_impurity_gain to measure feature importance.
            #self.feature_importance[node.feature_index] += impurity_gain
            #self.potential_feature_importance += impurity_gains
            return True
        return False


    def find_best_split_per_feature(self, parent, f, by): 
        """Find the split along feature f that minimises action_impurity for a parent node."""
        # Sort this node's subset along selected feature.
        indices_sorted = parent.indices[np.argsort(self.o[parent.indices,f])]
        # Initialise variables that will be iteratively modified.
        if by == 'action':    
            if self.classifier:
                per_sample_action_loss_left = np.zeros_like(parent.per_sample_action_loss)
                action_impurity_sum_left = 0.
                per_sample_action_loss_right = parent.per_sample_action_loss.copy()
                action_impurity_sum_right = parent.action_impurity_sum.copy()
            else:
                action_mean_left = 0.
                action_impurity_sum_left = 0.
                action_mean_right = parent.action_best.copy()
                action_impurity_sum_right = parent.action_impurity_sum.copy()
        elif by == 'value':
            value_mean_left = 0.
            value_impurity_sum_left = 0.
            value_mean_right = parent.value_mean.copy()
            value_impurity_sum_right = parent.value_impurity_sum.copy()
        # Iterate through thresholds.
        best_split = [None, 0, None]
        for num_left in range(self.min_samples_leaf_abs, parent.num_samples+1-self.min_samples_leaf_abs):
            i = indices_sorted[num_left-1]
            num_right = parent.num_samples - num_left
            o_f = self.o[i,f]
            o_f_next = self.o[indices_sorted[num_left],f]
            if by == 'action':            
                if self.classifier:
                    # Action impurity for classification: use (weighted) Gini.
                    w = self.w[i]
                    a = self.a[i]
                    loss_delta = w * self.pwl[a]
                    per_sample_action_loss_left += loss_delta
                    per_sample_action_loss_right -= loss_delta
                    action_impurity_sum_left += 2 * (w * per_sample_action_loss_left[a]) # NOTE: Assumes self.pwl is symmetric.
                    action_impurity_sum_right -= 2 * (w * per_sample_action_loss_right[a])
                    impurity_gain = parent.action_impurity - (((action_impurity_sum_left / num_left) + (action_impurity_sum_right / num_right)) / parent.num_samples) # Divide twice, multiply once.     
                else: 
                    # Action impurity for regression: use standard deviation. NOTE: Not variance!
                    # Incremental variance computation from http://datagenetics.com/blog/november22017/index.html.
                    # TODO: Make this calculation sensitive to sample weights.
                    a = self.a[i]
                    action_mean_left, action_impurity_sum_left = self.increment_mu_and_var_sum(action_mean_left, action_impurity_sum_left, a, num_left, 1)
                    action_mean_right, action_impurity_sum_right = self.increment_mu_and_var_sum(action_mean_right, action_impurity_sum_right, a, num_right, -1)     
                    # Square root turns into standard deviation.
                    impurity_gain = parent.action_impurity - ((math.sqrt(action_impurity_sum_left*num_left) + math.sqrt(max(0,action_impurity_sum_right)*num_right)) / parent.num_samples)
            elif by == 'value':
                # Value impurity: use standard deviation.
                g = self.g[i]
                value_mean_left, value_impurity_sum_left = self.increment_mu_and_var_sum(value_mean_left, value_impurity_sum_left, g, num_left, 1)
                value_mean_right, value_impurity_sum_right = self.increment_mu_and_var_sum(value_mean_right, value_impurity_sum_right, g, num_right, -1)
                # Square root turns into standard deviation.
                impurity_gain = parent.value_impurity - ((math.sqrt(value_impurity_sum_left*num_left) + math.sqrt(max(0,value_impurity_sum_right)*num_right)) / parent.num_samples)
            
            # Determine if this is the best split found so far.
            # Skip if this sample's feature value is the same as the next one.
            if o_f != o_f_next and impurity_gain > best_split[1] and impurity_gain > self.min_impurity_gain: 
                # Split at midpoint.
                best_split = [(o_f + o_f_next) / 2, impurity_gain, num_left]       
        #print(parent.value_impurity_sum, value_impurity_sum_left, value_impurity_sum_right) 
        return f, by, indices_sorted, best_split


    def increment_mu_and_var_sum(_, mu, var_sum, x, n, sign):
        """Incremental sum-of-variance computation from http://datagenetics.com/blog/november22017/index.html."""
        d_last = x - mu
        mu += sign * (d_last / n)
        d = x - mu
        var_sum += sign * (d_last * d)
        return mu, var_sum


# ===================================================================================================================
# METHODS FOR PRUNING.


# ===================================================================================================================
# METHODS FOR PREDICTION WITH UNSEEN SAMPLES.


    def predict(self, o, method='best', extra=None):
        """Predict actions for a set of observations."""
        shp = np.shape(o)
        if len(shp)==1 or shp[0]==1: return self.predict_one(o, method, extra)
        else: return [self.predict_one(oi, method, extra) for oi in o]

    
    def predict_one(self, o, method='best', extra=None):
        """Predict action for a single observation, optionally returning some additional information."""
        def recurse(node, o, method, extra):
            if not node.left: 
                if method == 'best': a = node.action_best
                elif method == 'sample': 
                    if self.classifier: 
                        # For classification, sample according to action probabilities.
                        a_i = np.random.choice(range(self.num_actions), p=node.action_probs)
                        a = self.action_order[a_i]
                    else: 
                        # For regression, pick a random member of the leaf.
                        a = self.a[np.random.choice(node.indices)]
                else: raise ValueError('Invalid prediction method.')
                if extra == None: return a
                else:
                    result = [a] 
                    if 'address' in extra: result.append(node.address)
                    if 'probabilities' in extra: 
                        if self.classifier: result.append(node.action_probs)
                        else: result.append(None)
                    # NOTE: value estimation just uses members of same leaf. 
                    # This has high variance if the population is small, so could perhaps do better
                    # by considering ancestor nodes (lower weight).
                    if 'value' in extra: 
                        if self.classifier: result.append(self.value_at_node(node, action=a_i)) # For classification can further condition on a.
                        else: result.append(self.value_at_node(node))
                    return tuple(result)
            if o[node.feature_index] < node.threshold: return recurse(node.left, o, method, extra)
            return recurse(node.right, o, method, extra)
        return recurse(self.tree, o, method, extra)

    
# ===================================================================================================================
# METHODS FOR TRAVERSING THE TREE GIVEN VARIOUS LOCATORS.


    def node(self, address):
        """Navigate to a node using its address."""
        node = self.tree
        for lr in address:
            if lr == 0:   assert node.left, 'Invalid address.'; node = node.left
            elif lr == 1: assert node.right, 'Invalid address.'; node = node.right
            else: raise ValueError('Invalid adddress.')
        return node
    
    
    def locate_sample(self, index):
        """Return the leaf node at which a sample is stored, and its address."""
        def recurse(node, index):
            if node.left and index in node.left.indices: 
                return recurse(node.left, index)
            if node.right and index in node.right.indices: 
                return recurse(node.right, index)
            return node, node.address
        return recurse(self.tree, index)


# ===================================================================================================================
# METHODS FOR WORKING WITH DYNAMIC TRAJECTORIES.


    def locate_prev_sample(self, index):
        """Run locate_sample to find the sample before the given one.""" 
        index_p = self.p[index]
        if index_p < 0: return None, (None, None)
        return index_p, self.locate_sample(index_n)
    def locate_next_sample(self, index):
        """Run locate_sample to find the sample after the given one.""" 
        index_n = self.n[index]
        if index_n < 0: return None, (None, None)
        return index_n, self.locate_sample(index_n)

    
    def sample_trajectory(self, index):
        """Return the full trajectory before and after a given sample."""
        before = []; index_p = index
        if self.p != []:
            while True: 
                index_p = self.p[index_p] 
                if index_p < 0: break # Negative index indicates the start of a trajectory.
                before.insert(0, index_p)
        after = [index]; index_n = index
        if self.n != []:
            while True: 
                index_n = self.n[index_n] 
                if index_n < 0: break # Negative index indicates the end of a trajectory.
                after.append(index_n)
        return before, after

    
    def compute_returns(self): 
        """Compute the returns for all samples in the dataset."""
        if self.r == []: self.g = []; return
        if not (self.p != [] and self.n != []): self.g = self.r; return
        self.g = np.zeros_like(self.r)
        for index in np.argwhere(self.n < 0): # Find indices of terminal observations.
            self.g[index] = self.r[index]
            index_p = self.p[index]
            while index_p >= 0:
                self.g[index_p] = self.r[index_p] + (self.gamma * self.g[index])
                index = index_p; index_p = self.p[index] 

    
    def leaf_transitions(self, leaf=None, address=None, action=None):
        """Given a leaf, find the next sample after each consistuent sample then group by the leaves at which these successor samples lie.
        Optionally condition this process on a given action action."""
        if leaf != None: assert address == None, 'Cannot specify both.'
        elif address != None: assert leaf == None, 'Cannot specify both.'; leaf = self.node(address)
        else: raise ValueError('Must pass either leaf or address.') 
        assert leaf.left == None and leaf.right == None, 'Node must be a leaf.'
        if action != None: 
            assert self.classifier, 'Can only condition on action in classification mode.'
            assert action in self.action_order, 'Action not recognised.'
            # Filter samples down to those with the specified action.
            indices = leaf.indices[self.a[leaf.indices]==action]
        else: indices = leaf.indices
        transitions = {}
        for index in indices:
            _, (_, address) = self.locate_next_sample(index)
            if address != None: address = tuple(address)
            if address in transitions: transitions[address].append(index)
            else: transitions[address] = [index]
        return transitions

    
    def leaf_transition_probs(self, leaf=None, address=None, action=None):
        """Convert the output of the previous method into probabilities."""
        transitions = self.leaf_transitions(leaf, address, action)
        n = sum(len(tr) for tr in transitions.values())
        return {l:len(tr)/n for l,tr in transitions.items()}

    
    def node_returns(self, node=None, address=None, action=None):
        """Given a node, find the return for each consistuent sample.
        Optionally condition this process on a given action action."""
        if node != None: assert address == None, 'Cannot specify both.'
        elif address != None: assert node == None, 'Cannot specify both.'; node = self.node(address)
        else: raise ValueError('Must pass either node or address.') 
        if action != None: 
            assert self.classifier, 'Can only condition on action in classification mode.'
            assert action in self.action_order, 'Action not recognised.'
            # Filter samples down to those with the specified action.
            indices = node.indices[self.a[node.indices]==action]
        else: indices = node.indices
        return self.g[indices]

    
    def value_at_node(self, node=None, address=None, action=None):
        """Convert the output of the previous method into a value by taking the mean."""
        if action == None: return node.value_mean
        return np.mean(self.node_returns(node, address, action))


# ===================================================================================================================
# METHODS FOR TREE DESCRIPTION AND VISUALISATION.


    def to_code(self, comment=False, alt_action_names=None, out_file=None): 
        """Print tree rules as an executable function definition.
        Adapted from https://stackoverflow.com/questions/20224526/how-to-extract-the-decision-rules-from-scikit-learn-decision-tree"""
        lines = []
        #lines.append("def tree({}):".format('ARGS'))#", ".join([f for f in self.feature_names])))
        def recurse(node, depth=0):
            indent = "    " * depth
            pred = self.action_order[node.action_best]
            if comment:
                conf = int(100 * node.action_probs[self.action_order.index(node.action_best)])
            if alt_action_names != None: pred = alt_action_names[pred]
            # Decision nodes.
            if node.left:
                feature = self.feature_names[node.feature_index]
                if comment: lines.append("{}if {} < {}: # depth = {}, best action = {}, confidence = {}%, weighted counts = {}, action_impurity = {}".format(indent, feature, node.threshold, depth, pred, conf, node.weighted_action_counts, node.action_impurity))
                else: lines.append("{}if {} < {}:".format(indent, feature, node.threshold))
                recurse(node.left, depth+1)
                if comment: lines.append("{}else: # if {} >= {}".format(indent, feature, node.threshold))
                else: lines.append("{}else:".format(indent))
                recurse(node.right, depth+1)
            # Leaf nodes.
            else:
                if comment: lines.append("{}return {} # confidence = {}%, weighted counts = {}, action_impurity = {}".format(indent, pred, conf, node.weighted_action_counts, node.action_impurity))
                else: lines.append("{}return {}".format(indent, pred))
        recurse(self.tree)

        # If no out file specified, just print.
        if out_file == None: 
           for line in lines: print(line)
        else: 
            with open(out_file+'.py', 'w', encoding='utf-8') as f:
                for l in lines: f.write(l+'\n')


    def to_dataframe(self, out_file=None):
        """Represent the leaf notes of the tree as rows of a Pandas dataframe."""
        data = []
        o_max = np.max(self.o, axis=0)
        o_min = np.min(self.o, axis=0)
        def recurse(node, partitions=[]):
            # Decision nodes.
            if node.left:
                recurse(node.left, partitions+[(node.feature_index, '<', node.threshold)])
                recurse(node.right, partitions+[(node.feature_index, '>', node.threshold)])
            # Leaf nodes.
            else:
                row = [tuple(node.address)]
                for f in range(self.num_features):
                    for sign in ['>','<']:
                        p_rel = [p for p in partitions if p[0] == f and p[1] == sign]
                        # The last partition for each (feature name, sign) pair is always the most restrictive.
                        if len(p_rel) > 0: val = p_rel[-1][2]
                        else: val = (o_max[f] if sign == '<' else o_min[f])  
                        row.append(val)
                if self.classifier: action_best = self.action_order[node.action_best]
                else: action_best = node.action_best
                count_fraction = node.num_samples / self.num_samples
                weight_sum = sum(self.w[node.indices])
                weight_fraction = weight_sum / self.w_sum
                if self.classifier: row += [action_best, node.action_counts, count_fraction, node.weighted_action_counts, node.action_impurity]
                else: row += [node.action_best, node.action_impurity]
                row += [node.value_mean, node.value_impurity, weight_fraction]
                data.append(row)
        recurse(self.tree)
        if self.classifier:
            action_columns = ['action_best','action_counts','count_fraction','weighted_action_counts','action_impurity']
        else: 
            action_columns = ['action_best','action_impurity']
        self.df = pd.DataFrame(data,columns=['address'] + [f+sign for f in self.feature_names for sign in [' >',' <']] + action_columns + ['value_mean','value_impurity','weight_fraction']).set_index('address')
        self.have_df = True
        # If no out file specified, just return.
        if out_file == None: return self.df
        else: self.df.to_csv(out_file+'.csv', index=False)

    
    def visualise(self, features, lims, ax=None, visualise=True, colour_by='action_best', action_colours=None, show_addresses=False, try_reuse_df=True):
        """Visualise the decision boundary across one or two features, possibly marginalising across all others."""
        n = len(features)
        assert n == len(lims) and n in (1,2)
        if n < self.num_features: marginalise = True
        else: marginalise = False
        if try_reuse_df and self.have_df: df = self.df
        else: df = self.to_dataframe()  
        regions = {}                  
        if not marginalise:
            # This is easy: can just use leaves directly.
            if n == 1: height = 1
            for address, leaf in df.iterrows():
                xy = []; outside_lims = False
                for i, (f, lim) in enumerate(zip(features, lims)):
                    f_min = leaf['{} >'.format(f)]
                    f_max = leaf['{} <'.format(f)]
                    # Ignore if leaf is outside lims.
                    if f_min >= lim[1] or f_max <= lim[0]: outside_lims = True; break
                    f_min = max(f_min, lim[0])
                    f_max = min(f_max, lim[1])
                    xy.append(f_min)
                    if i == 0: width = f_max - f_min 
                    else:      height = f_max - f_min 
                if outside_lims: continue
                if n == 1: xy.append(0)
                regions[address] = {'xy':xy, 'width':width, 'height':height, 'attribute':leaf[colour_by]}
        else:
            # Get all unique values mentioned in partitions for these features.
            f1 = features[0]
            p1 = np.unique(df[[f1+' >',f1+' <']].values) # Sorted by default.
            r1 = np.vstack((p1[:-1],p1[1:])).T # Ranges.
            if n == 2: 
                f2 = features[1]
                p2 = np.unique(df[[f2+' >',f2+' <']].values) 
                r2 = np.vstack((p2[:-1],p2[1:])).T
            else: r2 = [[None,None]]
            for m, ((min1, max1), (min2, max2)) in enumerate(np.array([[i,j] for i in r1 for j in r2])):
                if min1 >= lims[0][1] or max1 <= lims[0][0]: continue # Ignore if leaf is outside lims.
                min1 = max(min1, lims[0][0])
                max1 = min(max1, lims[0][1])
                width = max1 - min1
                if n == 1: 
                    xy = [min1, 0]
                    height = 1
                    # Get leaves that overlap with this region.
                    ol = df.loc[(df[f1+' >']<=min1) & (df[f1+' <']>=max1)]
                    # Compute the proportion of overlap for each leaf..
                    ol['overlap'] = ol.apply(lambda row: width / (row[f1+' <'] - row[f1+' >']), axis=1)
                else: 
                    if min2 >= lims[1][1] or max2 <= lims[1][0]: continue
                    min2 = max(min2, lims[1][0])
                    max2 = min(max2, lims[1][1])
                    xy = [min1, min2]
                    height = max2 - min2                
                    ol = df.loc[(df[f1+' >']<=min1) & (df[f1+' <']>=max1) & (df[f2+' >']<=min2) & (df[f2+' <']>=max2)]
                    ol['overlap'] = ol.apply(lambda row: (width / (row[f1+' <'] - row[f1+' >'])) * (height / (row[f2+' <'] - row[f2+' >'])), axis=1)
                # NOTE: Averaging process assumes uniform data distribution within leaves.
                if colour_by == 'action_best' and self.classifier:
                    # Special case for action with classification: discrete values.
                    attr_average = np.argmax(np.dot(np.vstack(ol['weighted_action_counts'].values).T, 
                                                              ol['overlap'].values.reshape(-1,1)))
                else:
                    # For all other cases, can just take weighted mean.
                    weight_sum = np.dot(ol['weight_fraction'].values, ol['overlap'].values)
                    ol['contrib'] = ol.apply(lambda row: (row['weight_fraction'] * row['overlap']) / weight_sum, axis=1)
                    attr_average = np.dot(ol[colour_by].values, ol['contrib'].values)  
                regions[m] = {'xy':xy, 'width':width, 'height':height, 'attribute':attr_average}           
        if visualise:
            from matplotlib import cm
            cmaps = {'action_best':(cm.viridis, 'viridis'), # For regression only.
                     'value_mean':(cm.RdBu, 'RdBu'),
                     'value_impurity':(cm.Reds, 'Reds'),
                     'action_impurity':(cm.Reds, 'Reds'),
                     'count_fraction':(cm.gray, 'gray'),
                     'weight_fraction':(cm.gray, 'gray'),
                     }
            assert colour_by in cmaps, 'Invalid colour_by attribute.'
            import matplotlib.pyplot as plt
            from matplotlib.patches import Rectangle
            if not ax: _, ax = plt.subplots()
            ax.set_title(f'Coloured by {colour_by}')
            ax.set_xlabel(features[0]); ax.set_xlim(lims[0])
            if n == 1: ax.set_ylim([0,1])    
            else: ax.set_ylabel(features[1]); ax.set_ylim(lims[1])    
            if colour_by == 'action_best' and self.classifier:
                assert action_colours != None, 'Specify colours for discrete actions.'
            else: 
                attr_max = max(df[colour_by])
                attr_min = min(df[colour_by])
                attr_range = attr_max - attr_min
                dummy = ax.imshow(np.array([[attr_min,attr_max]]), aspect='auto', cmap=cmaps[colour_by][1])
                dummy.set_visible(False)
                plt.colorbar(dummy, ax=ax) 
            for name, region in regions.items():
                if colour_by == 'action_best' and self.classifier:
                    colour = action_colours[region['attribute']]
                else: 
                    if attr_min == attr_max: colour = [1,1,1]
                    else: colour = cmaps[colour_by][0]((region['attribute'] - attr_min) / (attr_range))
                ax.add_patch(Rectangle(xy=region['xy'], width=region['width'], height=region['height'], 
                             facecolor=colour, edgecolor=None, zorder=-10))
                # Add leaf address.
                if not marginalise and show_addresses: 
                    ax.text(region['xy'][0]+region['width']/2, region['xy'][1]+region['height']/2, name, 
                            horizontalalignment='center', verticalalignment='center')
        return regions


# ===================================================================================================================
# NODE CLASS.


class Node():
    def __init__(self, 
                 address, 
                 num_samples, 
                 indices, 
                 ):
        """These are the basic attributes; more are added elsewhere."""
        self.address = address
        self.indices = indices
        self.num_samples = num_samples
        self.left = None
        self.right = None