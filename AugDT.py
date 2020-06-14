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
    def __init__(self,
                 classifier,                       # Whether actions are discrete or continuous.
                 feature_names = None,             # Assign alphanumeric names to features.
                 scale_features_by = 'range',      # Method used to determine feature scaling, or pre-computed vector of scales.
                 pairwise_action_loss =      {},   # (For classification) badness of each pairwise action error. If unspecified, assume uniform.
                 gamma =                     1     # Discount factor to use when considering reward.
                 ):   
        self.classifier = classifier
        if feature_names: self.feature_names = feature_names
        else:             self.feature_names = np.arange(self.num_features).astype(str) # Initialise to indices.
        self.num_features = len(feature_names)
        if type(scale_features_by) != str:
            assert len(scale_features_by) == self.num_features
            self.scale_features_by = 'given'
            # NOTE: Normalising using geometric mean.
            self.feature_scales = np.array(scale_features_by) / np.exp(np.mean(np.log(scale_features_by)))
        else: 
            self.scale_features_by = scale_features_by
            self.feature_scales = []
        self.pairwise_action_loss = pairwise_action_loss
        self.gamma = gamma
        self.tree = None
        self.cf = None 
        self.have_df = False 


# ===================================================================================================================
# COMPLETE GROWTH ALGORITHMS.


    def grow_depth_first(self, 
                         o, a, r=[], p=[], n=[], w=[],       # Dataset.
                         by='action',                        # Attribute to split by: action, value or both.
                         max_depth =                 np.inf, # Depth at which to stop splitting.  
                         min_samples_split =         2,      # Min samples at a node to consider splitting. 
                         min_weight_fraction_split = 0.,     # Min weight fraction at a node to consider splitting.
                         min_samples_leaf =          1,      # Min samples at a leaf to accept split.
                         min_impurity_gain =         0.,     # Min impurity gain to accept split.
                         stochastic_splits =         False,  # Whether to samples splits proportional to impurity gain. Otherwise deterministic argmax.
                         ):
        """
        Accept a complete dataset and grow a complete tree depth-first as in CART.
        """
        assert by in ('action','value','both')
        if by in ('value','both'): assert r != [], 'Need reward information to split by value.'
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split 
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_split = min_weight_fraction_split
        self.stochastic_splits = stochastic_splits
        self.min_impurity_gain = min_impurity_gain
        self.load_data(o, a, r, p, n, w, append=False)
        self.seed()
        def recurse(node, depth):
            if depth < self.max_depth and self.split(node, by):
                self.num_leaves += 1
                recurse(node.left, depth+1)
                recurse(node.right, depth+1)
        recurse(self.tree, 0)

    
    def grow_best_first(self, 
                        o, a, r=[], p=[], n=[], w=[],       # Dataset.
                        by='action',                        # Attribute to split by: action, value or both.
                        max_num_leaves =            np.inf, # Max number of leaves in tree.
                        min_samples_split =         2,      # Min samples at a node to consider splitting. 
                        min_weight_fraction_split = 0.,     # Min weight fraction at a node to consider splitting.
                        min_samples_leaf =          1,      # Min samples at a leaf to accept split.
                        min_impurity_gain =         0.,     # Min impurity gain to accept split.
                        stochastic_splits =         False,  # Whether to samples splits proportional to impurity gain. Otherwise deterministic argmax.
                        ):
        """
        Accept a complete dataset and grow a complete tree best-first.
        This is done by selecting the leaf with highest impurity_sum.
        """
        assert by in ('action','value','both')
        if by in ('value','both'): assert r != [], 'Need reward information to split by value.'
        self.max_num_leaves = max_num_leaves
        self.min_samples_split = min_samples_split 
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_split = min_weight_fraction_split
        self.stochastic_splits = stochastic_splits
        self.min_impurity_gain = min_impurity_gain
        self.load_data(o, a, r, p, n, w, append=False)
        self.seed()
        addresses = [()]
        impurities = [[self.tree.action_impurity_sum, self.tree.value_impurity_sum]]
        with tqdm(total=self.max_num_leaves) as pbar:
            while self.num_leaves < self.max_num_leaves and len(addresses) > 0:
                i_norm = np.array(impurities) 
                i_norm /= i_norm.max(axis=0)
                if by == 'action': best = np.argmax(i_norm[:,0])
                elif by == 'value': best = np.argmax(i_norm[:,1])
                # NOTE: For 'both', current approach is to sum normalised impurities and find argmax.
                elif by == 'both': best = np.argmax(i_norm.sum(axis=1))
                address = addresses.pop(best); impurities.pop(best)
                node = self.node(address)
                if self.split(node, by):
                    self.num_leaves += 1; pbar.update(1)
                    addresses.append(node.left.address)
                    impurities.append([node.left.action_impurity_sum, node.left.value_impurity_sum])
                    addresses.append(node.right.address)
                    impurities.append([node.right.action_impurity_sum, node.right.value_impurity_sum])

        
    """
    TODO: Other algorithms: 
        - Could also define 'best' as highest impurity gain, but this requires us to try splitting every node first!
        - Best first with active sampling (a la TREPAN).
        - Restructuring. 
    """


# ===================================================================================================================
# GENERIC METHODS FOR GROWTH.


    def load_data(self, o, a, r=[], p=[], n=[], w=[], append=False):
        """
        Complete training data has all of the following components, but only the first two are essential.
        *o = observations.
        *a = actions.
        r = rewards.
        p = index of preceding sample (negative = start of episode).
        n = index of successor sample (negative = end of episode).
        w = sample weight.
        """
        if append: raise Exception('Dataset appending not yet implemented.')

        # Store basic values.
        self.o, self.r, self.p, self.n = o, r, p, n
        self.num_samples, n_f = self.o.shape
        assert n_f == self.num_features, 'Observation size does not match feature_names.'
        assert self.num_samples == len(a) == len(r) == len(p) == len(n), 'All inputs must be the same length.'
        if w == []: self.w = np.ones(self.num_samples)
        else: self.w = w
        self.w_sum = self.w.sum()
        if self.classifier:
            # Define a canonical (alphanumeric) order for the actions so can work in terms of indices.
            self.action_names = sorted(list(set(a) | set(self.pairwise_action_loss)))
            self.num_actions = len(self.action_names)
            self.action_loss_to_matrix() 
            self.a = np.array([self.action_names.index(c) for c in a]) # Convert array into indices.
        else:
            self.a = a

        # Compute return for each sample.
        self.g = self.compute_returns(self.r, self.p, self.n)

        # Set up min samples for split / leaf if these were specified as ratios.
        if type(self.min_samples_split) == float: self.min_samples_split_abs = int(np.ceil(self.min_samples_split * self.num_samples))
        else: self.min_samples_split_abs = self.min_samples_split
        if type(self.min_samples_leaf) == float: self.min_samples_leaf_abs = int(np.ceil(self.min_samples_leaf * self.num_samples))
        else: self.min_samples_leaf_abs = self.min_samples_leaf

        # Automatically create feature scaling vector if not provided already.
        if self.feature_scales == []:
            if self.scale_features_by == 'range': 
                ranges = (np.max(self.o, axis=0) - np.min(self.o, axis=0))
            elif self.scale_features_by == 'percentiles':
                ranges = (np.percentile(self.o, 95, axis=0) - np.percentile(self.o, 5, axis=0))
            else: raise ValueError('Invalid feature scaling method.')
            inverse_ranges = 1 / ranges
            # NOTE: Normalising using geometric mean.
            self.feature_scales = inverse_ranges / np.exp(np.mean(np.log(inverse_ranges)))


    def action_loss_to_matrix(self):
        """Convert dictionary representation to a matrix for use in action_impurity calculations."""
        # If unspecified, use 1 - identity matrix.
        if self.pairwise_action_loss == {}: self.pwl = 1 - np.identity(self.num_actions)
        # Otherwise use values provided.
        else:
            self.pwl = np.zeros((self.num_actions,self.num_actions))
            for c,losses in self.pairwise_action_loss.items():
                ci = self.action_names.index(c)
                for cc,l in losses.items():
                    cci = self.action_names.index(cc)
                    # NOTE: Currently symmetric.
                    self.pwl[ci,cci] = l
                    self.pwl[cci,ci] = l


    def seed(self):
        """Initialise a new tree with its root node."""
        assert hasattr(self, 'o'), 'No data loaded.' 
        self.tree = self.new_leaf([], np.arange(self.num_samples))
        self.num_leaves = 1

    
    def new_leaf(self, address, indices):
        """Create a new leaf, computing attributes where required."""
        node = Node(address = tuple(address),
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
        # When doing 'best', we try both 'action' and 'value',
        # and choose whichever gives the highest impurity gain as a *ratio* of the parent impurity.
        candidate_splits = []; done_action = False; done_value = False
        if by in ('action','both') and node.action_impurity > 0: # Not necessary if zero impurity.
            done_action = True
            for f in range(self.num_features):
                candidate_splits.append(self.find_best_split_per_feature(node, f, 'action'))
                candidate_splits[-1][3].append(candidate_splits[-1][3][1] / node.action_impurity)      
        if by in ('value','both') and node.value_impurity > 0:
            done_value = True
            for f in range(self.num_features):
                candidate_splits.append(self.find_best_split_per_feature(node, f, 'value'))
                candidate_splits[-1][3].append(candidate_splits[-1][3][1] / node.value_impurity)
        # If beneficial split found on at least one feature...
        if sum([s[3][0] != None for s in candidate_splits]) > 0: 
            # Split quality = impurity gain / parent impurity (normalised to sum to one). 
            split_quality = [s[3][3] for s in candidate_splits]              
            # Choose one feature to split on.  
            if self.stochastic_splits:
                # Sample in proportion to relative impurity gain.
                chosen_split = np.random.choice(range(len(candidate_splits)), p=split_quality)
            else:
                # Deterministically choose the feature with greatest relative impurity gain.
                chosen_split = np.argmax(split_quality) # Ties broken by lowest index.                
            # Unpack information for this split and create child leaves.
            node.feature_index, node.split_by, indices_sorted, (node.threshold, _, split_index, _) = candidate_splits[chosen_split]            
            node.left = self.new_leaf(list(node.address)+[0], indices_sorted[:split_index])
            node.right = self.new_leaf(list(node.address)+[1], indices_sorted[split_index:])        
            # Store impurity gains, scaled by node.num_samples, to measure feature importance.
            node.feature_importance = np.zeros((4, self.num_features))
            if done_action:
                fi_action = np.array([s[3][1] for s in candidate_splits if s[1] == 'action']) * node.num_samples     
                node.feature_importance[2,:] = fi_action # Potential.
                if node.split_by == 'action': 
                    node.feature_importance[0,node.feature_index] = max(fi_action) # Realised.
            if done_value:
                fi_value = np.array([s[3][1] for s in candidate_splits if s[1] == 'value']) * node.num_samples
                node.feature_importance[3,:] = fi_value # Potential.
                if node.split_by == 'value': 
                    node.feature_importance[1,node.feature_index] = max(fi_value) # Realised.
            # Propagate importances back to all ancestors.
            address = node.address
            while address != ():
                ancestor, address = self.parent(address)
                ancestor.feature_importance += node.feature_importance
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


    def predict(self, o, method='best', attributes=['action'], use_action_names=True):
        """Predict actions for a set of observations, optionally returning some additional information."""
        # Test if just one sample has been provided.
        shp = np.shape(o)
        if len(shp)==1: o = [o]
        if type(attributes) == str: attributes = [attributes]
        R = {}
        for attr in attributes: R[attr] = []
        for oi in o:
            # Propagate each sample to its respective leaf.
            leaf = self.propagate(oi, self.tree)
            if 'action' in attributes:
                if method == 'best': a_i = leaf.action_best
                elif method == 'sample': 
                    if self.classifier: 
                        # For classification, sample according to action probabilities.
                        a_i = np.random.choice(range(self.num_actions), p=leaf.action_probs)                    
                    else: 
                        # For regression, pick a random member of the leaf.
                        a = self.a[np.random.choice(leaf.indices)]
                else: raise ValueError('Invalid prediction method.')
                # Convert to action names if applicable.
                if self.classifier and use_action_names: R['action'].append(self.action_names[a_i])
                else: R['action'].append(a_i)
            if 'address' in attributes: 
                R['address'].append(leaf.address)
            if 'uncertainty' in attributes: 
                if self.classifier: R['uncertainty'].append(leaf.action_probs)
                else: R['uncertainty'].append(leaf.action_impurity)
            if 'value_mean' in attributes:
                # NOTE: value estimation just uses members of same leaf. 
                # This has high variance if the population is small, so could perhaps do better
                # by considering ancestor nodes (lower weight).
                R['value_mean'].append(leaf.value_mean)
            if 'value_impurity' in attributes:
                R['value_impurity'].append(leaf.value_impurity)

        # Turn into numpy arrays.
        for attr in attributes:
            if attr == 'address': R[attr] = np.array(R[attr], dtype=object) # Allows variable length.
            else: R[attr] = np.array(R[attr]) 
        # Clean up what is returned if just one sample or attribute to include.
        #if len(o) == 1: R = {k:v[0] for k,v in R.items()}
        if len(attributes) == 1: R = R[attributes[0]]
        return R

    
    def propagate(self, o, node):
        """Propagate an unseen sample to a leaf node."""
        if node.left: 
            if o[node.feature_index] < node.threshold: return self.propagate(o, node.left)
            return self.propagate(o, node.right)  
        return node

    
# ===================================================================================================================
# METHODS FOR TRAVERSING THE TREE GIVEN VARIOUS LOCATORS.


    def node(self, address):
        """Navigate to a node using its address."""
        if address == None: return None
        node = self.tree
        for lr in address:
            if lr == 0:   assert node.left, 'Invalid address.'; node = node.left
            elif lr == 1: assert node.right, 'Invalid address.'; node = node.right
            else: raise ValueError('Invalid adddress.')
        return node

    
    def parent(self, address):
        """Navigate to a node's parent and return it and its address."""
        parent_address = address[:-1]
        return self.node(parent_address), parent_address
    
    
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

    
    def sample_episode(_, p, n, index):
        """Return the full episode before and after a given sample."""
        before = []; index_p = index
        if p != []:
            while True: 
                index_p = p[index_p] 
                if index_p < 0: break # Negative index indicates the start of a episode.
                before.insert(0, index_p)
        after = [index]; index_n = index
        if n != []:
            while True: 
                index_n = n[index_n] 
                if index_n < 0: break # Negative index indicates the end of a episode.
                after.append(index_n)
        return before, after

    
    def compute_returns(self, r, p, n): 
        """Compute returns for a set of samples."""
        if r == []: return []
        if not (p != [] and n != []): return r
        g = np.zeros_like(r)
        # Find indices of terminal observations.
        for index in np.argwhere((n < 0) | (np.arange(len(n)) == len(n)-1)): 
            g[index] = r[index]
            index_p = p[index]
            while index_p >= 0:
                g[index_p] = r[index_p] + (self.gamma * g[index])
                index = index_p; index_p = p[index] 
        return g

    
    def compute_returns_n_step_ordered_episode(self, r, p, n, steps):
        """Compute returns for an *ordered episode* of samples, 
        with a limit on the number of lookahead steps."""
        if steps == None: return self.compute_returns(r, p, n)
        if r == []: return []
        if (not (p != [] and n != [])) or steps == 1: return r
        assert steps > 0, 'Steps must be None or a positive integer.'
        # Precompute discount factors.
        discount = [1]
        for t in range(1, steps): discount.append(discount[-1]*self.gamma)
        discount = np.array(discount)
        # Iterate through samples.
        g = np.zeros_like(r)
        for index in range(len(g)):
            next_rewards = r[index:index+steps]
            g[index] = np.dot(next_rewards, discount[:len(next_rewards)])
        return g

    
    # def leaf_transitions_old(self, leaf=None, address=None, action=None):
    #     """Given a leaf, find the next sample after each consistuent sample then group by the leaves at which these successor samples lie.
    #     Optionally condition this process on a given action action."""
    #     if leaf != None: assert address == None, 'Cannot specify both.'
    #     elif address != None: assert leaf == None, 'Cannot specify both.'; leaf = self.node(address)
    #     else: raise ValueError('Must pass either leaf or address.') 
    #     assert leaf.left == None and leaf.right == None, 'Node must be a leaf.'
    #     if action != None: 
    #         assert self.classifier, 'Can only condition on action in classification mode.'
    #         assert action in self.action_names, 'Action not recognised.'
    #         # Filter samples down to those with the specified action.
    #         indices = leaf.indices[self.a[leaf.indices]==action]
    #     else: indices = leaf.indices
    #     transitions = {}
    #     for index in indices:
    #         _, (_, address) = self.locate_next_sample(index)
    #         if address in transitions: transitions[address].append(index)
    #         else: transitions[address] = [index]
    #     return transitions

    
    # def leaf_transition_probs_old(self, leaf=None, address=None, action=None):
    #     """Convert the output of the previous method into probabilities."""
    #     transitions = self.leaf_transitions(leaf, address, action)
    #     n = sum(len(tr) for tr in transitions.values())
    #     return {l:len(tr)/n for l,tr in transitions.items()}

    
    def leaf_transitions(self, leaf=None, address=None, previous_leaf=False, previous_address=False):
        """Given a leaf, find all constituent samples whose predecessors are not in this leaf.
        For each of these, step through the sequence of successors until this leaf is departed.
        Record the successor leaf (or None if terminal)."""
        if leaf != None: assert address == None, 'Cannot specify both.'
        elif address != None: assert leaf == None, 'Cannot specify both.'; leaf = self.node(address)
        else: raise ValueError('Must pass either leaf or address.') 
        if previous_leaf != False: assert previous_address == False, 'Cannot specify both.'
        elif previous_address != False: assert previous_leaf == False, 'Cannot specify both.'; previous_leaf = self.node(previous_address)
        assert leaf.left == None and leaf.right == None, 'Node must be a leaf.'
        if previous_leaf != False:
            # Filter samples down to those whose predecessor is in the previous leaf. 
            if previous_leaf == None: 
                first_indices = leaf.indices[self.p[leaf.indices] < 0]
            else: 
                assert previous_leaf.left == None and previous_leaf.right == None, 'Previous node must be a leaf.'
                first_indices = leaf.indices[np.isin(self.p[leaf.indices], previous_leaf.indices)]
        else:
            # Filter samples down to those whose predecessor is *not* in this leaf.
            first_indices = leaf.indices[np.isin(self.p[leaf.indices], leaf.indices, invert=True)]
        print(first_indices)
        transitions = {}
        for first_index in first_indices:
            index = first_index; n = 0
            while True:
                index, (next_leaf, address) = self.locate_next_sample(index)
                n += 1
                if next_leaf != leaf: break
            if address in transitions: transitions[address].append((first_index, n, self.g[first_index]))
            else: transitions[address] = [(first_index, n, self.g[first_index])]
        return transitions

    
    def leaf_transition_probs(self, leaf=None, address=None, previous_leaf=False, previous_address=False):
        """Convert the output of the previous method into probabilities."""
        transitions = self.leaf_transitions(leaf, address, previous_leaf, previous_address)
        n = sum(len(tr) for tr in transitions.values())
        durations = [np.mean([i[1] for i in tr]) for tr in transitions.values()]
        returns = [np.mean([i[2] for i in tr]) for tr in transitions.values()]
        return {l:(len(tr)/n, d, g) for (l,tr), d, g in zip(transitions.items(), durations, returns)}

    
    def node_returns(self, node=None, address=None, action=None):
        """Given a node, find the return for each consistuent sample.
        Optionally condition this process on a given action action."""
        if node != None: assert address == None, 'Cannot specify both.'
        elif address != None: assert node == None, 'Cannot specify both.'; node = self.node(address)
        else: raise ValueError('Must pass either node or address.') 
        if action != None: 
            assert self.classifier, 'Can only condition on action in classification mode.'
            assert action in self.action_names, 'Action not recognised.'
            # Filter samples down to those with the specified action.
            indices = node.indices[self.a[node.indices]==action]
        else: indices = node.indices
        return self.g[indices]

    
    def node_value(self, node=None, address=None, action=None):
        """Convert the output of the previous method into a value by taking the mean."""
        if action == None: return node.value_mean
        return np.mean(self.node_returns(node, address, action))


# ===================================================================================================================
# METHODS FOR WORKING WITH COUNTERFACTUAL DATA


    def cf_load_data(self, o, a, r, p, n, regret_steps=np.inf, append=True):
        """
        Counterfactual data looks a lot like training data, 
        but is assumed to originate from a policy other than the target one,
        so must be kept separate.
        """
        # assert self.tree != None, 'Must have already grown tree.'
        assert len(o) == len(a) == len(r) == len(p) == len(n), 'All inputs must be the same length.'
        if self.cf == None: self.cf = Counterfactual(self)
        
        # Compute return for each new sample.
        g = self.compute_returns(r, p, n)

        # Initialise regret array.
        regret = np.empty_like(g); regret[:] = np.nan

        # Store the exploratory data, appending if applicable.
        if self.classifier: a = np.array([self.action_names.index(c) for c in a]) # Convert array into indices.
        if append == False or self.cf.o == []:
            num_cf_prev = 0
            self.cf.o, self.cf.a, self.cf.r, self.cf.p, self.cf.n, self.cf.g = o, a, r, p, n, g
            self.cf.regret = regret
        else:
            num_cf_prev = len(self.cf.o)
            if self.cf.regret_steps != regret_steps:
                assert self.cf.regret_steps == None, "Can't use different values of regret_steps in an appended dataset; recompute first."
            self.cf.o = np.vstack((self.cf.o, o))
            self.cf.a = np.hstack((self.cf.a, a))
            self.cf.r = np.hstack((self.cf.r, r))
            self.cf.p = np.hstack((self.cf.p, p))
            self.cf.n = np.hstack((self.cf.n, n))
            self.cf.g = np.hstack((self.cf.g, g))
            self.cf.regret = np.hstack((self.cf.regret, regret))

        # Compute regret for each new sample. This is done on a per-episode basis then concatenated.
        for index in np.argwhere(np.logical_and(np.arange(len(p)) >= num_cf_prev, p < 0)):
            ep_indices, regret = self.cf_regret_trajectory(index[0], regret_steps)
            self.cf.regret[ep_indices] = regret
        self.cf.regret_steps = regret_steps


    def cf_regret_trajectory(self, index, steps=np.inf):
        """
        Compute n-step regrets vs the estimated value function
        for a trajectory of counterfactual samples starting at index.
        """
        assert self.tree != None, 'Must have grown tree to meaningfully analyse.'
        # Retrieve all successive samples in the counterfactual episode.
        _, indices = self.sample_episode(self.cf.p, self.cf.n, index)
        o = self.cf.o[indices]
        r = self.cf.r[indices]
        p = self.cf.p[indices]
        n = self.cf.n[indices]
        # Verify steps.
        num_samples = len(r)
        if steps >= num_samples: steps = num_samples-1
        else: assert steps > 0, 'Steps must be None or a positive integer.'
        # Compute n-step returns.
        g = self.compute_returns_n_step_ordered_episode(r, p, n, steps)[:num_samples-steps]
        # Use the extant tree to get value predictions.
        v = self.predict(o, attributes='value_mean')        
        # Compute regret = v - (n-step return + discounted value of state in n-steps' time).
        regret = v[:num_samples-steps] - (g + (v[steps:] * (self.gamma ** steps)))
        return indices[:num_samples-steps], regret

    
    def cf_regret_all(self, steps):
        """
        Compute n-step regrets vs the estimated value function
        for all samples in the counterfactual dataset.
        """
        self.cf.regret = np.empty_like(self.cf.regret); self.cf.regret[:] = np.nan
        for index in np.argwhere(self.cf.p < 0):
            ep_indices, regret = self.cf_regret_trajectory(index[0], steps)
            self.cf.regret[ep_indices] = regret
        self.cf.regret_steps = steps

        
    def cf_node_criticality():



# ===================================================================================================================
# METHODS FOR TREE DESCRIPTION AND VISUALISATION.


    def to_code(self, comment=False, alt_action_names=None, out_file=None): 
        """
        Print tree rules as an executable function definition. Adapted from:
        https://stackoverflow.com/questions/20224526/how-to-extract-the-decision-rules-from-scikit-learn-decision-tree
        """
        lines = []
        #lines.append("def tree({}):".format('ARGS'))#", ".join([f for f in self.feature_names])))
        def recurse(node, depth=0):
            indent = "    " * depth
            pred = self.action_names[node.action_best]
            if comment:
                conf = int(100 * node.action_probs[self.action_names.index(node.action_best)])
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


    def to_dataframe(self, include_internal=True, out_file=None):
        """
        Represent the nodes of the tree as rows of a Pandas dataframe.
        """
        data = []
        o_max = np.max(self.o, axis=0)
        o_min = np.min(self.o, axis=0)
        def recurse(node, partitions=[]):
            if node.left == None or include_internal:
                row = [node.address, len(node.address), ('internal' if node.left else 'leaf')]
                ranges = []; 
                for f in range(self.num_features):
                    ranges.append([])
                    for sign in ['>','<']:
                        p_rel = [p for p in partitions if p[0] == f and p[1] == sign]
                        # The last partition for each (feature name, sign) pair is always the most restrictive.
                        if len(p_rel) > 0: val = p_rel[-1][2]
                        else: val = (o_max[f] if sign == '<' else o_min[f])  
                        ranges[-1].append(val); row.append(val)
                if self.classifier: action_best = self.action_names[node.action_best]
                else: action_best = node.action_best
                count_fraction = node.num_samples / self.num_samples
                weight_sum = sum(self.w[node.indices])
                weight_fraction = weight_sum / self.w_sum
                # Volume of a leaf = product of feature ranges, scaled by self.feature_scales
                ranges = np.array(ranges)
                volume = np.prod((ranges[:,1] - ranges[:,0]) * self.feature_scales)
                sample_density = node.num_samples / volume
                weight_density = weight_sum / volume
                if self.classifier: row += [action_best, node.action_counts, count_fraction, node.weighted_action_counts, node.action_impurity]
                else: row += [node.action_best, node.action_impurity]
                row += [node.value_mean, node.value_impurity, node.num_samples, weight_sum, weight_fraction, volume, sample_density, weight_density]
                data.append(row)
            # For decision nodes, recurse to children.
            if node.left:
                recurse(node.left, partitions+[(node.feature_index, '<', node.threshold)])
                recurse(node.right, partitions+[(node.feature_index, '>', node.threshold)])
        recurse(self.tree)
        if self.classifier:
            action_columns = ['action_best','action_counts','count_fraction','weighted_action_counts','action_impurity']
        else: 
            action_columns = ['action_best','action_impurity']
        self.df = pd.DataFrame(data,columns=['address','depth','type'] + [f+sign for f in self.feature_names for sign in [' >',' <']] 
                                                               + action_columns 
                                                               + ['value_mean','value_impurity','num_samples','weight_sum','weight_fraction','volume','sample_density','weight_density']
                              ).set_index('address')
        self.have_df = True
        # If no out file specified, just return.
        if out_file == None: return self.df
        else: self.df.to_csv(out_file+'.csv', index=False)

    
    def visualise(self, features, lims=[], axes=[], visualise=True, attributes=['action_best'], action_colours=None, edge_colour=None, show_addresses=False, try_reuse_df=True):
        """
        Visualise attributes across one or two features, 
        possibly marginalising across all others.
        """
        n = len(features)
        assert n in (1,2), 'Can only plot in 1 or 2 dimensions.'
        if try_reuse_df and self.have_df: 
            df = self.df.loc[self.df['type']=='leaf'] # Only care about leaves.
        else: 
            df = self.to_dataframe()
            df = df.loc[df['type']=='leaf']  
        if lims == []:
            # If lims not specified, use global lims across dataset.
            fi = [self.feature_names.index(f) for f in features]
            lims = np.vstack((np.min(self.o[:,fi], axis=0), np.max(self.o[:,fi], axis=0))).T
        from matplotlib import cm
        cmaps = {'action_best':(cm.RdBu, 'RdBu'), # For regression only.
                 'value_mean':(cm.RdBu, 'RdBu'),
                 'value_impurity':(cm.Reds, 'Reds'),
                 'action_impurity':(cm.Reds, 'Reds'),
                 'sample_density':(cm.gray, 'gray'),
                 'weight_density':(cm.gray, 'gray'),
                 }
        if type(attributes) == str: attributes = [attributes]
        for attr in attributes: assert attr in cmaps, 'Invalid attribute.'
        if n < self.num_features: marginalise = True
        else: marginalise = False
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
                regions[address] = {'xy':xy, 'width':width, 'height':height}
                for attr in attributes: regions[address][attr] = leaf[attr]   
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
            for m, ((min1, max1), (min2, max2)) in enumerate(tqdm(np.array([[i,j] for i in r1 for j in r2]))):
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
                regions[m] = {'xy':xy, 'width':width, 'height':height}     
                # NOTE: Averaging process assumes uniform data distribution within leaves.
                for attr in attributes:
                    if attr == 'action_best' and self.classifier:
                        # Special case for action with classification: discrete values.
                        regions[m][attr] = np.argmax(np.dot(np.vstack(ol['weighted_action_counts'].values).T, 
                                                                      ol['overlap'].values.reshape(-1,1)))
                    else: 
                        if attr in ('sample_density','weight_density'): normaliser = 'volume' # Another special case for densities.
                        else:                                           normaliser = 'weight_sum'
                        # Take contribution-weighted mean.
                        norm_sum = np.dot(ol[normaliser].values, ol['overlap'].values)
                        ol['contrib'] = ol.apply(lambda row: (row[normaliser] * row['overlap']) / norm_sum, axis=1)
                        regions[m][attr] = np.dot(ol[attr].values, ol['contrib'].values)                    
        if visualise:
            import matplotlib.pyplot as plt
            from matplotlib.patches import Rectangle
            for a, attr in enumerate(attributes):
                if axes != []: 
                    if len(attributes) == 1: ax = axes
                    else: ax = axes[a]
                else: _, ax = plt.subplots()
                ax.set_title(f'Coloured by {attr}')
                ax.set_xlabel(features[0]); ax.set_xlim(lims[0])
                if n == 1: ax.set_ylim([0,1]); ax.set_yticks([])  
                else: ax.set_ylabel(features[1]); ax.set_ylim(lims[1])    
                if attr == 'action_best' and self.classifier:
                    assert action_colours != None, 'Specify colours for discrete actions.'
                else: 
                    # NOTE: Scaling by percentiles.
                    attr_list = [r[attr] for r in regions.values()]
                    upper_perc = np.percentile(attr_list, 95)
                    lower_perc = np.percentile(attr_list, 5)
                    perc_range = upper_perc - lower_perc
                    dummy = ax.imshow(np.array([[lower_perc,upper_perc]]), aspect='auto', cmap=cmaps[attr][1])
                    dummy.set_visible(False)
                    plt.colorbar(dummy, ax=ax, orientation=('horizontal' if n==1 else 'vertical')) 
                for name, region in regions.items():
                    if attr == 'action_best' and self.classifier:
                        colour = action_colours[region[attr]]
                    else: 
                        if lower_perc == upper_perc: colour = [1,1,1]
                        else: colour = cmaps[attr][0]((region[attr] - lower_perc) / (perc_range))
                    ax.add_patch(Rectangle(xy=region['xy'], width=region['width'], height=region['height'], 
                                facecolor=colour, edgecolor=edge_colour, zorder=-10))
                    # Add leaf address.
                    if not marginalise and show_addresses: 
                        ax.text(region['xy'][0]+region['width']/2, region['xy'][1]+region['height']/2, name, 
                                horizontalalignment='center', verticalalignment='center')
        return regions


    def cf_scatter_regret(self, features, indices=None, lims=[], ax=None):
        """
        Create a scatter plot showing all counterfactual samples,
        coloured by their regret.
        """
        import matplotlib.pyplot as plt
        from matplotlib import cm
        if not ax: _, ax = plt.subplots()
        assert len(features) == 2, 'Can only plot in 1 or 2 dimensions.'
        # If indices not specified, use all.
        if indices == None: indices = np.arange(len(self.cf.o))
        indices = indices[~np.isnan(self.cf.regret[indices])] # Remove NaNs.
        o, regret = self.cf.o[indices], self.cf.regret[indices]
        # NOTE: Scaling by percentiles.
        upper_perc = np.percentile(regret, 95)
        lower_perc = np.percentile(regret, 5)
        perc_range = upper_perc - lower_perc
        # Set lims.
        if lims == []:
            # If lims not specified, use global lims across dataset.
            fi = [self.feature_names.index(f) for f in features]
            lims = np.vstack((np.min(self.o[:,fi], axis=0), np.max(self.o[:,fi], axis=0))).T
        ax.set_xlim(lims[0]); ax.set_ylim(lims[1])    
        # Define colours.
        dummy = ax.imshow(np.array([[lower_perc,upper_perc]]), aspect='auto', cmap='Reds')
        dummy.set_visible(False)
        plt.colorbar(dummy, ax=ax, orientation='horizontal') 
        colours = cm.Reds((regret - lower_perc) / perc_range)
        # Plot.
        ax.scatter(o[:,0], o[:,1], s=0.5, color=colours)
        return ax


# ===================================================================================================================
# NODE CLASS.


class Node():
    def __init__(self, 
                 address, 
                 num_samples, 
                 indices, 
                 ):
        # These are the basic attributes; more are added elsewhere.
        self.address = address
        self.indices = indices
        self.num_samples = num_samples
        self.left = None
        self.right = None

    
# ===================================================================================================================
# CLASS FOR HOLDING EXPLORATORY DATA.


class Counterfactual(): 
    def __init__(self, model): 
        self.o = [] # Initially dataset is empty.
        self.regret_steps = None