"""A variant of the CART tree induction algorithm, 
augmented with features that are particular 
to the task of RL policy distillation.
"""

import numpy as np
import math
import pandas as pd
pd.options.mode.chained_assignment = None
from scipy.spatial import minkowski_distance
from tqdm import tqdm
import matplotlib
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

class AugDT:
    def __init__(self,
                 classifier,                   # Whether actions are discrete or continuous.
                 feature_names = None,         # Assign alphanumeric names to features.
                 scale_features_by = 'range',  # Method used to determine feature scaling, or pre-computed vector of scales.
                 scale_derivatives_by = 'std', # Method used to determine derivative scaling.
                 pairwise_action_loss = {},    # (For classification) badness of each pairwise action error. If unspecified, assume uniform.
                 gamma = 1                     # Discount factor to use when considering reward.
                 ):   
        self.classifier = classifier
        if feature_names: self.feature_names = feature_names
        else:             self.feature_names = np.arange(self.num_features).astype(str) # Initialise to indices.
        self.num_features = len(feature_names)
        if type(scale_features_by) != str:
            assert len(scale_features_by) == self.num_features
            self.scale_features_by = 'given'
            self.feature_scales = np.array(scale_features_by) 
        else: 
            self.scale_features_by = scale_features_by
            self.feature_scales = []
        assert scale_derivatives_by == 'std', 'Only standard deviation implemented.'
        self.scale_derivatives_by = scale_derivatives_by
        self.derivative_scales = []
        self.min_samples_split = 2; self.min_samples_leaf = 1
        self.pairwise_action_loss = pairwise_action_loss
        self.gamma = gamma
        self.tree = None
        self.cf = None 
        self.have_highlights = False
        self.have_df = False 


# ===================================================================================================================
# COMPLETE GROWTH ALGORITHMS.

    # TODO: min_samples_split and min_weight_fraction_split not used!
    
    # TODO: Master grow() method to prevent duplications.

    def grow_depth_first(self, 
                         o, a, r=[], p=[], n=[], w=[],           # Dataset.
                         split_by =                  'weighted', # Attribute to split by: action, value or both.
                         gain_relative_to =          'root',     # Whether to normalise gains relative to parent or root.
                         value_weight =              0,          # Weight of value impurity (if by = 'weighted').
                         max_depth =                 np.inf,     # Depth at which to stop splitting.  
                         min_samples_split =         2,          # Min samples at a node to consider splitting. 
                         min_weight_fraction_split = 0,          # Min weight fraction at a node to consider splitting.
                         min_samples_leaf =          1,          # Min samples at a leaf to accept split.
                         min_split_quality =         0,          # Min relative impurity gain to accept split.
                         stochastic_splits =         False,      # Whether to samples splits proportional to impurity gain. Otherwise deterministic argmax.
                         ):
        """
        Accept a complete dataset and grow a complete tree depth-first as in CART.
        """
        assert split_by in ('action','value','pick','weighted')
        if split_by in ('value','pick','weighted'): assert r != [], 'Need reward information to split by value.'
        if split_by == 'weighted': assert value_weight >= 0 and value_weight <= 1
        elif split_by == 'action': value_weight = 0
        elif split_by == 'value': value_weight = 1
        assert gain_relative_to in ('parent','root')     
        self.split_by = split_by
        self.gain_relative_to = gain_relative_to
        self.imp_weights = np.array([1-value_weight, value_weight])   
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split 
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_split = min_weight_fraction_split
        self.stochastic_splits = stochastic_splits
        self.min_split_quality = min_split_quality
        self.load_data(o, a, r, p, n, w)
        self.seed()
        def recurse(node, depth):
            if depth < self.max_depth and self.split(node):
                recurse(node.left, depth+1)
                recurse(node.right, depth+1)
        print('Growing...')
        recurse(self.tree, 0)
        # List all the leaf integers.
        self.leaf_nints = self.get_leaf_nints()
        # Compute leaf transition probabilities, both marginal and conditional.
        print('Computing transition probabilities...')
        self.compute_all_leaf_transition_probs()

    
    def grow_best_first(self, 
                        o, a, r=[], p=[], n=[], w=[],           # Dataset.
                        split_by =                  'weighted', # Attribute to split by: action, value or both.
                        gain_relative_to =          'root',     # Whether to normalise gains relative to parent or root.
                        value_weight =              0.5,        # Weight of value impurity (if by='weighted').
                        max_num_leaves =            np.inf,     #
                        min_samples_split =         2,          #
                        min_weight_fraction_split = 0,          # Min weight fraction at a node to consider splitting.
                        min_samples_leaf =          1,          # Min samples at a leaf to accept split.
                        min_split_quality =         0,          # Min relative impurity gain to accept split.
                        stochastic_splits =         False,      # Whether to samples splits proportional to impurity gain. Otherwise deterministic argmax.
                        ):
        """
        Accept a complete dataset and grow a complete tree best-first.
        This is done by selecting the leaf with highest impurity_sum.
        """
        assert split_by in ('action','value','pick','weighted')
        if split_by in ('value','pick','weighted'): assert r != [], 'Need reward information to split by value.'
        if split_by == 'weighted': assert value_weight >= 0 and value_weight <= 1
        elif split_by == 'action': value_weight = 0
        elif split_by == 'value': value_weight = 1
        assert gain_relative_to in ('parent','root')  
        self.split_by = split_by
        self.gain_relative_to = gain_relative_to
        self.imp_weights = np.array([1-value_weight, value_weight])   
        self.max_num_leaves = max_num_leaves
        self.min_samples_split = min_samples_split 
        self.min_weight_fraction_split = min_weight_fraction_split
        self.min_samples_leaf = min_samples_leaf
        self.stochastic_splits = stochastic_splits
        self.min_split_quality = min_split_quality
        self.load_data(o, a, r, p, n, w)
        self.seed()
        self.untried_leaf_nints = [1]
        self.leaf_impurities = [[self.tree.action_impurity_sum, self.tree.value_impurity_sum]]
        print('Growing...')
        with tqdm(total=self.max_num_leaves) as pbar:
            pbar.update(1)
            while self.num_leaves < self.max_num_leaves and len(self.untried_leaf_nints) > 0:
                self.split_next_best(pbar)
        # List all the leaf integers.
        self.leaf_nints = self.get_leaf_nints()
        # Compute leaf transition probabilities, both marginal and conditional.
        print('Computing transition probabilities...')
        self.compute_all_leaf_transition_probs()
                

    def split_next_best(self, pbar=None):
        """
        Find and split the single most impurity leaf in the tree.
        """
        assert self.tree, 'Must have started growth process already.'
        if self.leaf_impurities == []: return False
        imp = np.array(self.leaf_impurities)
        root_imp = np.array([self.tree.action_impurity_sum, self.tree.value_impurity_sum])
        imp_norm = imp / root_imp
        if self.split_by == 'action': best = np.argmax(imp_norm[:,0])
        elif self.split_by == 'value': best = np.argmax(imp_norm[:,1])
        # NOTE: For split_by='pick', current approach is to sum normalised impurities and find argmax.
        elif self.split_by == 'pick': best = np.argmax(imp_norm.sum(axis=1))
        # NOTE: For split_by='weighted', take weighted sum instead. 
        elif self.split_by == 'weighted': best = np.argmax(np.inner(imp_norm, self.imp_weights))
        nint = self.untried_leaf_nints.pop(best)
        imp = self.leaf_impurities.pop(best)
        node = self.node(nint)
        if self.split(node):
            if pbar: pbar.update(1)
            self.untried_leaf_nints.append(node.left.nint)
            self.leaf_impurities.append([node.left.action_impurity_sum, node.left.value_impurity_sum])
            self.untried_leaf_nints.append(node.right.nint)
            self.leaf_impurities.append([node.right.action_impurity_sum, node.right.value_impurity_sum])
            return True
        # If can't make a split, recurse to try the next best.
        else: return self.split_next_best()

    """
    TODO: Other algorithms: 
        - Could also define 'best' as highest impurity gain, but this requires us to try splitting every node first!
        - Best first with active sampling (a la TREPAN).
        - Restructuring. 
    """


# ===================================================================================================================
# METHODS FOR GROWTH.


    def load_data(self, o, a, r=[], p=[], n=[], w=[], append=False):
        """
        Complete training data has all of the following components, but only the first two are essential.
        *o = observations.
        *a = actions.
        r = rewards.
        p = index of preceding sample (-1 = start of episode).
        n = index of successor sample (-1 = end of episode).
        w = sample weight.
        """
        if append: raise Exception('Dataset appending not yet implemented.')

        # Store basic values.
        self.o, self.r, self.p, self.n = o, r, p, n
        self.num_samples, n_f = self.o.shape
        assert n_f == self.num_features, 'Observation size does not match feature_names.'
        assert self.num_samples == len(a) == len(r) == len(p) == len(n), 'All inputs must be the same length.'
        assert min(p) == min(n) >= -1, 'Episode start/end must be denoted by index of -1.'
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
        self.global_feature_lims = np.vstack((np.min(self.o, axis=0), np.max(self.o, axis=0))).T
        # Placeholder for storing the leaf at which each sample resides.
        # NOTE: Additional zero at the end handles episode termination cases.
        self.nint = np.zeros(self.num_samples+1).astype(int)
        # Placeholder for storing the next leaf after each sample.
        self.next_nint = np.zeros(self.num_samples).astype(int)
        # Compute return for each sample.
        self.g = self.get_returns(self.r, self.p, self.n)
        # Compute time derivatives of features for each sample.
        self.d = self.get_derivatives(self.o, self.p, self.n)
        # Set up min samples for split / leaf if these were specified as ratios.
        if type(self.min_samples_split) == float: self.min_samples_split_abs = int(np.ceil(self.min_samples_split * self.num_samples))
        else: self.min_samples_split_abs = self.min_samples_split
        if type(self.min_samples_leaf) == float: self.min_samples_leaf_abs = int(np.ceil(self.min_samples_leaf * self.num_samples))
        else: self.min_samples_leaf_abs = self.min_samples_leaf

        # Automatically create feature scaling vectors if not provided already.
        if self.feature_scales == []:
            if self.scale_features_by == 'range': 
                ranges = self.global_feature_lims[:,1] - self.global_feature_lims[:,0]
            elif self.scale_features_by == 'percentiles':
                ranges = (np.percentile(self.o, 95, axis=0) - np.percentile(self.o, 5, axis=0))
            else: raise ValueError('Invalid feature scaling method.')
            self.feature_scales = max(ranges) / ranges
        if self.derivative_scales == []:
            if self.scale_derivatives_by == 'std':
                ranges = np.nanstd(self.d, axis=0)
            self.derivative_scales = max(ranges) / ranges

        # Z-normalise feature derivatives. These are used for impurity calculations.
        self.d_norm = (self.d - np.nanmean(self.d, axis=0)) * self.derivative_scales
        
            

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
        nint = bits_to_int(address)
        node = Node(nint = nint,
                    indices = indices,
                    num_samples = len(indices),
                    )
        # Store this leaf as the site of each sample.
        self.nint[indices] = nint
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
        # Feature derivative attributes.
        if self.d != []:
            node.derivative_mean = np.nanmean(self.d[indices], axis=0)
            node.d_norm_mean = np.nanmean(self.d_norm[indices], axis=0)
            var = np.nanvar(self.d_norm[indices], axis=0)
            node.d_norm_impurity_sum = var * node.num_samples
            # NOTE: Taking sum of variance in z-normalised derivatives, and using standard deviation!
            node.d_norm_impurity = math.sqrt(var.sum()) 
        else: node.d_norm_mean = 0; node.d_norm_impurity = 0 
        # Placeholder for counterfactual samples and criticality.
        node.cf_indices = []
        # NOTE: Initialising criticalities at zero.
        # This has an effect on both visualisation and HIGHLIGHTS in the case of leaves
        # where no counterfactual data ends up.
        node.criticality_mean = 0; node.criticality_impurity = 0
        return node


    def split(self, node):
        """
        Split a leaf node to minimise some measure of impurity.
        """
        assert node.left == None, 'Not a leaf node.'
        # Check whether able to skip consideration of action or value entirely.
        if (node.action_impurity > 0) and (self.split_by == 'pick' or self.imp_weights[0] > 0): do_action = True
        else: do_action = False
        if (node.value_impurity > 0) and (self.split_by == 'pick' or self.imp_weights[1] > 0): do_value = True
        else: do_value = False
        if (not do_action) and (not do_value): return False
        # Get values to normalise action and value gains by.
        if self.gain_relative_to == 'parent':
            action_gain_normaliser = node.action_impurity
            value_gain_normaliser = node.value_impurity
        elif self.gain_relative_to == 'root':
            action_gain_normaliser = self.tree.action_impurity
            value_gain_normaliser = self.tree.value_impurity
        # Iterate through features and find best split(s) for each.
        candidate_splits = []
        for f in range(self.num_features):
            candidate_splits += self.split_feature(
            node, f, action_gain_normaliser, value_gain_normaliser, do_action, do_value)
        # If beneficial split found on at least one feature...
        if sum([s[3][0] != None for s in candidate_splits]) > 0: 
            split_quality = [s[3][2] for s in candidate_splits]              
            # Choose one feature to split on.  
            if self.stochastic_splits:
                # Sample in proportion to relative impurity gain.
                chosen_split = np.random.choice(range(len(candidate_splits)), p=split_quality)
            else:
                # Deterministically choose the feature with greatest relative impurity gain.
                chosen_split = np.argmax(split_quality) # Ties broken by lowest index.       
            # Unpack information for this split and create child leaves.
            node.feature_index, node.split_by, indices_sorted, (node.threshold, split_index, _, _, _) = candidate_splits[chosen_split]  
            address = int_to_bits(node.nint)
            node.left = self.new_leaf(list(address)+[0], indices_sorted[:split_index])
            node.right = self.new_leaf(list(address)+[1], indices_sorted[split_index:])        
            self.num_leaves += 1
            # Store impurity gains, scaled by node.num_samples, to measure feature importance.
            node.feature_importance = np.zeros((4, self.num_features))
            if do_action:
                fi_action = np.array([s[3][3] for s in candidate_splits if s[1] in ('action','weighted')]) * node.num_samples     
                node.feature_importance[2,:] = fi_action # Potential.
                node.feature_importance[0,node.feature_index] = max(fi_action) # Realised.
            if do_value:
                fi_value = np.array([s[3][4] for s in candidate_splits if s[1] in ('value','weighted')]) * node.num_samples     
                node.feature_importance[3,:] = fi_value # Potential.
                node.feature_importance[1,node.feature_index] = max(fi_value) # Realised.
            # Back-propagate importances to all ancestors.
            while address != ():
                ancestor, address = self.parent(address)
                ancestor.feature_importance += node.feature_importance
            return True
        return False


    # TODO: Make variance calculations sensitive to sample weights.
    def split_feature(self, parent, f, action_gain_normaliser, value_gain_normaliser, do_action, do_value): 
        """
        Find the split(s) along feature f that minimise(s) the impurity of the children.
        Impurity gain could be measured for action or value individually, or as a weighted sum.
        """
        # Sort this node's indices along selected feature.
        indices_sorted = parent.indices[np.argsort(self.o[parent.indices,f])]
        # Initialise variables that will be iteratively modified.
        if do_action:    
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
        else: action_impurity_gain = action_rel_impurity_gain = 0
        if do_value: 
            value_mean_left = 0.
            value_impurity_sum_left = 0.
            value_mean_right = parent.value_mean.copy()
            value_impurity_sum_right = parent.value_impurity_sum.copy()
        else: value_impurity_gain = value_rel_impurity_gain = 0

        # Iterate through thresholds.
        if self.split_by == 'pick': best_split = [[f,'action',indices_sorted,[None,None,0,0,0]],[f,'value',indices_sorted,[None,None,0,0,0]]]
        else: best_split = [[f,self.split_by,indices_sorted,[None,None,0,0,0]]]
        for num_left in range(self.min_samples_leaf_abs, parent.num_samples+1-self.min_samples_leaf_abs):
            i = indices_sorted[num_left-1]
            num_right = parent.num_samples - num_left
            
            if do_action:             
                if self.classifier:
                    # Action impurity for classification: use (weighted) Gini.
                    w = self.w[i]
                    a = self.a[i]
                    loss_delta = w * self.pwl[a]
                    per_sample_action_loss_left += loss_delta
                    per_sample_action_loss_right -= loss_delta
                    action_impurity_sum_left += 2 * (w * per_sample_action_loss_left[a]) # NOTE: Assumes self.pwl is symmetric.
                    action_impurity_sum_right -= 2 * (w * per_sample_action_loss_right[a])
                    action_impurity_gain = parent.action_impurity - (((action_impurity_sum_left / num_left) + (action_impurity_sum_right / num_right)) / parent.num_samples) # Divide twice, multiply once.     
    
                else: 
                    # Action impurity for regression: use standard deviation. NOTE: Not variance!
                    # Incremental variance computation from http://datagenetics.com/blog/november22017/index.html.
                    a = self.a[i]
                    action_mean_left, action_impurity_sum_left = self.increment_mu_and_var_sum(action_mean_left, action_impurity_sum_left, a, num_left, 1)
                    action_mean_right, action_impurity_sum_right = self.increment_mu_and_var_sum(action_mean_right, action_impurity_sum_right, a, num_right, -1)     
                    # Square root turns into standard deviation.
                    action_impurity_gain = parent.action_impurity - ((math.sqrt(action_impurity_sum_left*num_left) + math.sqrt(max(0,action_impurity_sum_right)*num_right)) / parent.num_samples)  
            
            if do_value:
                # Value impurity: use standard deviation.
                g = self.g[i]
                value_mean_left, value_impurity_sum_left = self.increment_mu_and_var_sum(value_mean_left, value_impurity_sum_left, g, num_left, 1)
                value_mean_right, value_impurity_sum_right = self.increment_mu_and_var_sum(value_mean_right, value_impurity_sum_right, g, num_right, -1)
                # Square root turns into standard deviation.
                value_impurity_gain = parent.value_impurity - ((math.sqrt(value_impurity_sum_left*num_left) + math.sqrt(max(0,value_impurity_sum_right)*num_right)) / parent.num_samples)

            # Skip if this sample's feature value is the same as the next one.
            o_f = self.o[i,f]
            o_f_next = self.o[indices_sorted[num_left],f]
            if o_f == o_f_next: continue

            if do_action: action_rel_impurity_gain = action_impurity_gain / action_gain_normaliser
            if do_value: value_rel_impurity_gain = value_impurity_gain  / value_gain_normaliser
           
            if self.split_by == 'pick':
                # Look at action and value individually.
                if action_rel_impurity_gain > self.min_split_quality and action_rel_impurity_gain > best_split[0][3][2]: 
                    best_split[0][3] = [(o_f + o_f_next) / 2, num_left, action_rel_impurity_gain, action_impurity_gain, value_impurity_gain]  
                if value_rel_impurity_gain > self.min_split_quality and value_rel_impurity_gain > best_split[1][3][2]: 
                    best_split[1][3] = [(o_f + o_f_next) / 2, num_left, value_rel_impurity_gain, action_impurity_gain, value_impurity_gain]  
            else: 
                # Calculate combined relative gain as weighted sum.
                combined_rel_impurity_gain = (self.imp_weights[0] * action_rel_impurity_gain) + (self.imp_weights[1] * value_rel_impurity_gain)
                if combined_rel_impurity_gain > self.min_split_quality and combined_rel_impurity_gain > best_split[0][3][2]: 
                    best_split[0][3] = [(o_f + o_f_next) / 2, num_left, combined_rel_impurity_gain, action_impurity_gain, value_impurity_gain]  

        return best_split


    def increment_mu_and_var_sum(_, mu, var_sum, x, n, sign):
        """Incremental sum-of-variance computation from http://datagenetics.com/blog/november22017/index.html."""
        d_last = x - mu
        mu += sign * (d_last / n)
        d = x - mu
        var_sum += sign * (d_last * d)
        return mu, var_sum


    def split_transitions(self, node):
        """
        Split a leaf node to minimise transition impurity.
        """
        assert node.left == None, 'Not a leaf node.'
        # Iterate through features and find best split for each.
        candidate_splits = []
        for f in range(self.num_features):
            candidate_splits += self.split_feature_transitions_v1(node, f)
        # If beneficial split found on at least one feature...
        if sum([s[3][0] != None for s in candidate_splits]) > 0: 
            split_quality = [s[3][2] for s in candidate_splits]
            print(node.nint, node.transition_impurity_sum, 'SPLIT', split_quality)
            # Deterministically choose the feature with greatest relative impurity gain.
            chosen_split = np.argmax(split_quality) # Ties broken by lowest index. 
            # Unpack information for this split and create child leaves.
            node.feature_index, node.split_by, indices_sorted, (node.threshold, split_index, _) = candidate_splits[chosen_split]  
            address = int_to_bits(node.nint)
            node.left = self.new_leaf(list(address)+[0], indices_sorted[:split_index])
            node.right = self.new_leaf(list(address)+[1], indices_sorted[split_index:])     
            self.num_leaves += 1
            # Remove transition information from parent because no longer relevent. 
            del node.next_nints; del node.next_nint_counts; del node.transition_impurity_sum   
            return True
        print(node.nint, 'NO SPLIT')
        return False
    # Version 1 = analyse at the level of trajectories,
    # and factor in new transitions created by the partition itself. 
    def compute_transition_impurities_v1(self, nints=[]):
        """
        Compute the impurity of the transitions from each leaf using the Gini coefficient.
        Here, only the last sample in each sequence is used.
        Optionally specify a subset of leaf integers; otherwise will do all.
        """
        if nints == []: nints = self.leaf_nints
        for nint in nints:
            leaf = self.node(nint)
            # Filter samples down to those whose successor is *not* in this leaf.
            last_indices = leaf.indices[np.nonzero(self.nint[self.n[leaf.indices]] != nint)]
            # Get the count for each next leaf.
            leaf.next_nints, leaf.next_nint_counts = np.unique(self.nint[self.n[last_indices]], return_counts=True)
            leaf.next_nints = list(leaf.next_nints)
            # Compute the Gini impurity of these counts.
            counts_sum = np.sum(leaf.next_nint_counts)
            leaf.transition_impurity = 1 - (np.sum(leaf.next_nint_counts**2) / counts_sum**2)
            leaf.transition_impurity_sum = leaf.transition_impurity * leaf.num_samples 

        # Return the leaf integers, sorted by transition_impurity_sum.
        return sorted({nint:self.node(nint).transition_impurity_sum for nint in self.leaf_nints}.items(), key = lambda x: x[1]) 
    def split_feature_transitions_v1(self, parent, f): 
        """
        xxx.
        """
        # Sort this node's indices along selected feature.
        indices_sorted = parent.indices[np.argsort(self.o[parent.indices,f])]
        # Initialise variables that will be iteratively modified.
        next_nint_counts_left = np.zeros(len(parent.next_nint_counts)+1).astype(int) # Need extra index to account for transition between the children.
        left_indices = set()
        next_nint_counts_right = np.append(parent.next_nint_counts.copy(), 0)
        right_indices = set(indices_sorted)
        
        # Iterate through thresholds.
        best_split = [[f,'transition',indices_sorted,[None,None,0]]]
        for num_left in range(self.min_samples_leaf_abs, parent.num_samples+1-self.min_samples_leaf_abs):
            i = indices_sorted[num_left-1]
            left_indices.add(i)
            right_indices.remove(i) 

            # Use Boolean tests to see if we have added or removed a transition.
            p_in_left = self.p[i] in left_indices
            p_in_right = self.p[i] in right_indices
            n_in_left = self.n[i] in left_indices
            n_in_right = self.n[i] in right_indices

            #print()
            #print(i, self.o[i], self.n[i], self.nint[self.n[i]])
            #print(p_in_left, p_in_right, n_in_left, n_in_right)

            if p_in_left:
                if not(n_in_right): next_nint_counts_left[-1] -= 1 # Remove left-to-right transition.
            else:
                if n_in_right: next_nint_counts_left[-1] += 1 # Add left-to-right transition.
            if n_in_left:
                if not(p_in_right): next_nint_counts_right[-1] -= 1 # Remove right-to-left transition.
            else:
                if p_in_right: next_nint_counts_right[-1] += 1 # Add right-to-left transition.
            #if not(n_in_left):
                # Transfer external transition from left to right.
                if not(n_in_right):
                    next_nint_index = parent.next_nints.index(self.nint[self.n[i]])
                    next_nint_counts_left[next_nint_index] += 1
                    next_nint_counts_right[next_nint_index] -= 1
            
            # Skip if this sample's feature value is the same as the next one.
            o_f = self.o[i,f]
            o_f_next = self.o[indices_sorted[num_left],f]
            if o_f == o_f_next: continue
            
            # Compute gain in impurity sum.
            # counts_sum_left = np.sum(next_nint_counts_left)
            # counts_sum_right = np.sum(next_nint_counts_right)
            # transition_impurity_sum_gain = parent.transition_impurity_sum \
            #                              - ((counts_sum_left - (np.sum(next_nint_counts_left**2) / counts_sum_left**2)) \
            #                              + (counts_sum_right - (np.sum(next_nint_counts_right**2) / counts_sum_right**2)))
            num_right = parent.num_samples - num_left
            transition_impurity_gain = parent.transition_impurity \
                                         - (((num_left * (1 - (np.sum(next_nint_counts_left**2) / np.sum(next_nint_counts_left)**2))) \
                                         + (num_right * (1 - (np.sum(next_nint_counts_right**2) / np.sum(next_nint_counts_right)**2)))) \
                                         / parent.num_samples)

            # print(parent.next_nint_counts, next_nint_counts_left, next_nint_counts_right, transition_impurity_sum_gain)

            # Check if best split so far.
            if transition_impurity_gain > best_split[0][3][2]:
                best_split[0][3] = [(o_f + o_f_next) / 2, num_left, transition_impurity_gain]
        
        # print(f, best_split[0][3])

        return best_split
    # Version 2 = analyse at the level of samples,
    # and ignore new transitions. This is more conventional classification.
    def compute_transition_impurities_v2(self, nints=[]): 
        """
        Compute the impurity of the transitions from each leaf using the Gini coefficient.
        Here, every sample in the leaf is used.
        Optionally specify a subset of leaf integers; otherwise will do all.
        """
        if nints == []: nints = self.leaf_nints
        for nint in nints:
            leaf = self.node(nint)
            # Recompute next_nints for all samples in this leaf.
            first_indices = leaf.indices[np.nonzero(self.nint[self.p[leaf.indices]] != nint)]            
            for index in first_indices:
                next_nint, sequence = self.get_next_nint(index)
                self.next_nint[sequence] = next_nint
            # Get the count for each next leaf.
            leaf.next_nints, leaf.next_nint_counts = np.unique(self.next_nint[leaf.indices], return_counts=True)
            leaf.next_nints = list(leaf.next_nints)
            # Compute the Gini impurity of these counts.
            leaf.transition_impurity_sum = leaf.num_samples - (np.sum(leaf.next_nint_counts**2) / leaf.num_samples)
            leaf.transition_impurity = leaf.transition_impurity_sum / leaf.num_samples 

        # Return the leaf integers, sorted by transition_impurity_sum.
        return sorted({nint:self.node(nint).transition_impurity_sum for nint in self.leaf_nints}.items(), key = lambda x: x[1]) 
    def split_feature_transitions_v2(self, parent, f): 
        """
        xxx.
        """
        # Sort this node's indices along selected feature.
        indices_sorted = parent.indices[np.argsort(self.o[parent.indices,f])]
        # Initialise variables that will be iteratively modified.
        next_nint_counts_left = np.zeros_like(parent.next_nint_counts) # Need extra index to account for transition between the children.
        next_nint_counts_right = parent.next_nint_counts.copy()
        
        # Iterate through thresholds.
        best_split = [[f,'transition',indices_sorted,[None,None,0]]]
        for num_left in range(self.min_samples_leaf_abs, parent.num_samples+1-self.min_samples_leaf_abs):
            i = indices_sorted[num_left-1]
            num_right = parent.num_samples - num_left

            # Transfer transition from left to right.
            next_nint_index = parent.next_nints.index(self.next_nint[i])
            next_nint_counts_left[next_nint_index] += 1
            next_nint_counts_right[next_nint_index] -= 1

            # Skip if this sample's feature value is the same as the next one.
            o_f = self.o[i,f]
            o_f_next = self.o[indices_sorted[num_left],f]
            if o_f == o_f_next: continue
            
            # Compute gain in impurity.
            transition_impurity_gain = parent.transition_impurity \
                                         - (((num_left - (np.sum(next_nint_counts_left**2) / num_left)) \
                                         + (num_right - (np.sum(next_nint_counts_right**2) / num_right))) \
                                         / parent.num_samples)

            # print(parent.next_nint_counts, (o_f + o_f_next) / 2, next_nint_counts_left, next_nint_counts_right, transition_impurity_sum_gain)

            # Check if best split so far.
            if transition_impurity_gain > best_split[0][3][2]:
                best_split[0][3] = [(o_f + o_f_next) / 2, num_left, transition_impurity_gain]
        
        # print(f, best_split[0][3])

        return best_split
    # Version 3 = analyse at the level of trajectories, and ignore new transitions.
    def compute_transition_impurities_v3(self, nints=[]): 
        """
        Compute the impurity of the transitions from each leaf using the Gini coefficient.
        Here, only the last sample in each sequence is used but we normalise by the total number of samples.
        Optionally specify a subset of leaf integers; otherwise will do all.
        """
        if nints == []: nints = self.leaf_nints
        for nint in nints:
            leaf = self.node(nint)
            # Filter samples down to those whose successor is *not* in this leaf.
            last_indices = leaf.indices[np.nonzero(self.nint[self.n[leaf.indices]] != nint)]
            # Get the count for each next leaf.
            leaf.next_nints, leaf.next_nint_counts = np.unique(self.nint[self.n[last_indices]], return_counts=True)
            leaf.next_nints = list(leaf.next_nints)
            # Compute the Gini impurity of these counts.
            counts_sum = np.sum(leaf.next_nint_counts)
            leaf.transition_impurity = 1 - (np.sum(leaf.next_nint_counts**2) / counts_sum**2)
            leaf.transition_impurity_sum = leaf.transition_impurity * leaf.num_samples 

        # Return the leaf integers, sorted by transition_impurity_sum.
        return sorted({nint:self.node(nint).transition_impurity_sum for nint in self.leaf_nints}.items(), key = lambda x: x[1]) 
    def split_feature_transitions_v3(self, parent, f): 
        """
        xxx.
        """
        # Sort this node's indices along selected feature.
        indices_sorted = parent.indices[np.argsort(self.o[parent.indices,f])]
        # Initialise variables that will be iteratively modified.
        next_nint_counts_left = np.zeros_like(parent.next_nint_counts) # Need extra index to account for transition between the children.
        next_nint_counts_right = parent.next_nint_counts.copy()
        
        # Iterate through thresholds.
        best_split = [[f,'transition',indices_sorted,[None,None,0]]]
        for num_left in range(self.min_samples_leaf_abs, parent.num_samples+1-self.min_samples_leaf_abs):
            i = indices_sorted[num_left-1]
            num_right = parent.num_samples - num_left

            # Transfer transition from left to right.
            if self.nint[self.n[i]] != parent.nint:
                next_nint_index = parent.next_nints.index(self.nint[self.n[i]])
                next_nint_counts_left[next_nint_index] += 1
                next_nint_counts_right[next_nint_index] -= 1

            # Skip if this sample's feature value is the same as the next one.
            o_f = self.o[i,f]
            o_f_next = self.o[indices_sorted[num_left],f]
            if o_f == o_f_next: continue
            
            # Compute gain in impurity.
            # transition_impurity_gain = parent.transition_impurity \
            #                              - (((num_left * (1 - (np.sum(next_nint_counts_left**2) / np.sum(next_nint_counts_left)**2))) \
            #                              + (num_right * (1 - (np.sum(next_nint_counts_right**2) / np.sum(next_nint_counts_right)**2)))) \
            #                              / parent.num_samples)

            transition_impurity_gain = parent.transition_impurity \
                                         - (((num_left * (1 - (np.sum(next_nint_counts_left**2) / np.sum(next_nint_counts_left)**2))) \
                                         + (num_right * (1 - (np.sum(next_nint_counts_right**2) / np.sum(next_nint_counts_right)**2)))) \
                                         / parent.num_samples)


            # print(parent.nint, self.nint[self.n[i]], num_left, num_right)
            # print(parent.next_nints, parent.next_nint_counts, (o_f + o_f_next) / 2, next_nint_counts_left, next_nint_counts_right, transition_impurity_gain)

            # Check if best split so far.
            if transition_impurity_gain > best_split[0][3][2]:
                best_split[0][3] = [(o_f + o_f_next) / 2, num_left, transition_impurity_gain]
        
        # print(f, best_split[0][3])

        return best_split


    def split_derivatives(self, node):
        """
        Split a leaf node to minimise feature derivative impurity.
        """
        assert node.left == None, 'Not a leaf node.'
        # Iterate through features and find best split for each.
        candidate_splits = []
        for f in range(self.num_features):
            candidate_splits += self.split_feature_derivatives(node, f)
        # If beneficial split found on at least one feature...
        if sum([s[3][0] != None for s in candidate_splits]) > 0: 
            split_quality = [s[3][2] for s in candidate_splits]
            print(node.nint, node.d_norm_impurity, 'SPLIT', split_quality)
            # Deterministically choose the feature with greatest relative impurity gain.
            chosen_split = np.argmax(split_quality) # Ties broken by lowest index. 
            # Unpack information for this split and create child leaves.
            node.feature_index, node.split_by, indices_sorted, (node.threshold, split_index, _) = candidate_splits[chosen_split]  
            address = int_to_bits(node.nint)
            node.left = self.new_leaf(list(address)+[0], indices_sorted[:split_index])
            node.right = self.new_leaf(list(address)+[1], indices_sorted[split_index:])     
            self.num_leaves += 1  
            return True
        print(node.nint, 'NO SPLIT')
        return False

    
    def split_feature_derivatives(self, parent, f):
        """
        Find the split(s) along feature f that minimise(s) the derivative impurity of the children.
        """
        # Sort this node's indices along selected feature.
        indices_sorted = parent.indices[np.argsort(self.o[parent.indices,f])]
        # Initialise variables that will be iteratively modified.
        d_norm_mean_left = 0.
        d_norm_impurity_sum_left = 0.
        d_norm_mean_right = parent.d_norm_mean.copy()
        d_norm_impurity_sum_right = parent.d_norm_impurity_sum.copy()

        # Iterate through thresholds.
        best_split = [[f,'derivative',indices_sorted,[None,None,0]]]
        for num_left in range(self.min_samples_leaf_abs, parent.num_samples+1-self.min_samples_leaf_abs):
            i = indices_sorted[num_left-1]
            num_right = parent.num_samples - num_left

            # Skip if derivative not defined for this sample.
            d_norm = self.d_norm[i]
            if np.isnan(np.sum(d_norm)): continue

            d_norm_mean_left, d_norm_impurity_sum_left = self.increment_mu_and_var_sum(d_norm_mean_left, d_norm_impurity_sum_left, d_norm, num_left, 1)
            d_norm_mean_right, d_norm_impurity_sum_right = self.increment_mu_and_var_sum(d_norm_mean_right, d_norm_impurity_sum_right, d_norm, num_right, -1)

            # Skip if this sample's feature value is the same as the next one.
            o_f = self.o[i,f]
            o_f_next = self.o[indices_sorted[num_left],f]
            if o_f == o_f_next: continue

            # Square root turns into standard deviation.
            d_norm_impurity_gain = parent.d_norm_impurity \
                                 - ((math.sqrt(d_norm_impurity_sum_left.sum()*num_left) \
                                 + math.sqrt(max(0,d_norm_impurity_sum_right.sum())*num_right)) 
                                 / parent.num_samples)

            # print(d_norm, parent.d_norm_impurity_sum, d_norm_impurity_sum_left, d_norm_impurity_sum_right, d_norm_impurity_gain)

            # Check if best split so far.
            if d_norm_impurity_gain > best_split[0][3][2]:
                best_split[0][3] = [(o_f + o_f_next) / 2, num_left, d_norm_impurity_gain]

        return best_split


# ===================================================================================================================
# METHODS FOR PRUNING.


    # TODO: Bring over.
    

# ===================================================================================================================
# METHODS FOR PREDICTION AND SCORING.


    def predict(self, o, method='best', attributes=['action'], use_action_names=True):
        """
        Predict actions for a set of observations, 
        optionally returning some additional information.
        """
        # Test if just one sample has been provided.
        shp = o.shape
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
            if 'nint' in attributes: 
                R['nint'].append(leaf.nint)
            if 'uncertainty' in attributes: 
                if self.classifier: R['uncertainty'].append(leaf.action_probs)
                else: R['uncertainty'].append(leaf.action_impurity)
            if 'value' in attributes:
                # NOTE: value/criticality estimation just uses members of same leaf. 
                # This has high variance if the population is small, so could perhaps do better
                # by considering ancestor nodes (lower weight).
                R['value'].append(leaf.value_mean)
            if 'value_impurity' in attributes:
                R['value_impurity'].append(leaf.value_impurity)
            if 'criticality' in attributes:
                R['criticality'].append(leaf.criticality_mean)
            if 'criticality_impurity' in attributes:
                R['criticality_impurity'].append(leaf.criticality_impurity)
        # Turn into numpy arrays.
        for attr in attributes:
            #if attr == 'address': R[attr] = np.array(R[attr], dtype=object) # Allows variable length.
            #else: 
            R[attr] = np.array(R[attr]) 
        # Clean up what is returned if just one sample or attribute to include.
        if len(attributes) == 1: R = R[attributes[0]]
        return R

    
    def propagate(self, o, node):
        """
        Propagate an unseen sample to a leaf node.
        """
        if node.left: 
            if o[node.feature_index] < node.threshold: return self.propagate(o, node.left)
            return self.propagate(o, node.right)  
        return node


    def score(self, o, a=[], g=[], action_metric=None, value_metric='mse'):
        """
        Score action and/or return prediction performance on a test set.
        """
        if action_metric == None: 
            if self.classifier: action_metric = 'error_rate'
            else: action_metric = 'mse'
        R = self.predict(o, attributes=['action','value_mean'])
        if a == []: action_score = None
        else: 
            if action_metric == 'error_rate':
                action_score = np.linalg.norm(R['action']- a, ord=0) / len(a)
            elif action_metric == 'mae':
                action_score = np.linalg.norm(R['action']- a, ord=1) / len(a)
            elif action_metric == 'mse':
                action_score = np.linalg.norm(R['action']- a, ord=2) / len(a)
        if g == []: value_score = None
        else:
            if value_metric == 'mae':
                value_score = np.linalg.norm(R['value_mean']- g, ord=1) / len(g)
            elif value_metric == 'mse':
                value_score = np.linalg.norm(R['value_mean']- g, ord=1) / len(g)

        return action_score, value_score
    
    
# ===================================================================================================================
# METHODS FOR TRAVERSING THE TREE GIVEN VARIOUS LOCATORS.


    def get_leaf_nints(self):
        """
        List the integers of all leaves in the tree.
        """
        def recurse(node):
            if node.left:
                return recurse(node.left) + recurse(node.right) 
            return [node.nint]
        return recurse(self.tree)


    def node(self, identifier):
        """
        Navigate to a node using its address or integer.
        """
        if identifier == None: return None
        elif type(identifier) == int: identifier = int_to_bits(identifier)
        node = self.tree
        for lr in identifier:
            if lr == 0:   assert node.left, 'Invalid identifier.'; node = node.left
            elif lr == 1: assert node.right, 'Invalid identifier.'; node = node.right
            else: raise ValueError('Invalid identifier.')
        return node

    
    def parent(self, address):
        """
        Navigate to a node's parent and return it and its address (not integer!)
        """
        parent_address = address[:-1]
        return self.node(parent_address), parent_address


# ===================================================================================================================
# METHODS FOR WORKING WITH DYNAMIC TRAJECTORIES.

    
    def sample_episode(_, p, n, index):
        """
        Return the full episode before and after a given sample.
        """
        before = []; index_p = index
        if p != []:
            while True: 
                index_p = p[index_p] 
                if index_p == -1: break # Index = -1 indicates the start of a episode.
                before.insert(0, index_p)
        after = [index]; index_n = index
        if n != []:
            while True: 
                index_n = n[index_n] 
                if index_n == -1: break # Index = -1 indicates the end of a episode.
                after.append(index_n)
        return np.array(before), np.array(after)

    
    def split_into_episodes(self, p, n):
        """
        Given lists of predecessor / successor relations (p, n), 
        split the indices by episode and put in temporal order.
        """
        return [self.sample_episode([], n, index[0])[1] 
                for index in np.argwhere(p == -1)]

            
    def get_returns(self, r, p, n): 
        """
        Compute returns for a set of samples.
        """
        if r == []: return []
        if not (p != [] and n != []): return r
        g = np.zeros_like(r)
        # Find indices of terminal observations.
        for index in np.argwhere((n == -1) | (np.arange(len(n)) == len(n)-1)): 
            g[index] = r[index]
            index_p = p[index]
            while index_p >= 0:
                g[index_p] = r[index_p] + (self.gamma * g[index])
                index = index_p; index_p = p[index] 
        return g

    
    def get_returns_n_step_ordered_episode(self, r, p, n, steps):
        """
        Compute returns for an *ordered episode* of samples, 
        with a limit on the number of lookahead steps.
        """
        if steps == None: return self.get_returns(r, p, n)
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

    
    def get_derivatives(self, o, p, n):
        """
        Compute the time derivatives of observation features for a set of samples.
        - Use only the next sample (n) because practically interested in what comes after.
        - Where n = -1 (terminal state), return NaNs.
        """
        # if p == [] or n == []: return []
        # # Assemble the arrays required.
        # o_p = self.o[p]; o_n = self.o[n]
        # # Calculate differences to previous and next samples.
        # nans = np.full_like(o[0], np.nan)
        # diffs = np.array([[(o[i] - o_p[i]) if p[i] != -1 else nans for i in range(len(o))],
        #                   [(o_n[i] - o[i]) if n[i] != -1 else nans for i in range(len(o))]])
        # return np.nanmean(diffs, axis=0)
        o_n = self.o[n]; nans = np.full_like(o[0], np.nan)
        return np.array([(o_n[i] - o[i]) if n[i] != -1 else nans for i in range(len(o))])


    def get_next_nint(self, index):
        """
        Given a sample, find the next leaf encountered in the successor sequence.
        Also return the sequence of samples upto this time.
        """
        nint = self.nint[index]; sequence = [] 
        while True:
            sequence.append(index); index = self.n[index]; next_nint = self.nint[index]
            if next_nint != nint: break
        return next_nint, sequence
    
    
    def get_leaf_transitions(self, nint):
        """
        Given a leaf integer, find all constituent samples whose predecessors are not in this leaf.
        For each of these, step through the sequence of successors until this leaf is departed.
        Record both the previous and next leaf (or 0 if terminal).
        """
        leaf = self.node(nint)
        assert leaf.left == None and leaf.right == None, 'Node must be a leaf.'
        # Filter samples down to those whose predecessor is *not* in this leaf.
        first_indices = leaf.indices[np.nonzero(self.nint[self.p[leaf.indices]] != nint)]
        prev, nxt, both = {}, {}, {}
        for index in first_indices:
            # Get the integer for the previous leaf.
            nint_p = self.nint[self.p[index]]
            # Get the integer for the next leaf, and the sequence of successors up to that time.
            next_nint, sequence = self.get_next_nint(index)
            # Store information about this sequence with previous, next and both together.
            info = [len(sequence), self.g[index]] # Sequence length and return from first sample.
            if nint_p in prev: prev[nint_p].append(info)
            else: prev[nint_p] = [info]
            if next_nint in nxt: nxt[next_nint].append(info)
            else: nxt[next_nint] = [info]
            pn = (nint_p, next_nint) 
            if pn in both: both[pn].append(info)
            else: both[pn] = [info]
        # Convert dictionary entries into numpy arrays.
        prev = {k:np.array(v) for k,v in prev.items()}
        nxt = {k:np.array(v) for k,v in nxt.items()}
        both = {k:np.array(v) for k,v in both.items()}
        return prev, nxt, both, len(first_indices)

    
    def get_leaf_transition_probs(self, nint):
        """
        Convert the output of the get_leaf_transitions method into probabilities:
            - Previous/next leaf marginal.
            - Previous/next conditional (on next/previous).
        """
        prev, nxt, both, n = self.get_leaf_transitions(nint)
        P = {'prev':{},'next':{}}
        # Function for processing sequences into the right form.
        f = lambda seqs, n : [len(seqs)/n, len(seqs)] + list(np.mean(seqs, axis=0))
        # For marginals, normalise by total number of sequences.
        P['prev']['marginal'] = {k:f(v, n) for k,v in prev.items()}
        P['next']['marginal'] = {k:f(v, n) for k,v in nxt.items()}
        # For conditionals, normalise by number of sequences matching condition.
        for cond, (_,n,_,_) in P['next']['marginal'].items():
            P['prev'][cond] = {k[0]:f(v, n) for k,v in both.items()
                               if k[1] == cond} # Filter with condition.
        for cond, (_,n,_,_) in P['prev']['marginal'].items():
            P['next'][cond] = {k[1]:f(v, n) for k,v in both.items()
                               if k[0] == cond} # Filter with condition.
        return P

    
    def compute_all_leaf_transition_probs(self):
        """
        Run the get_leaf_transition_probs method for all leaves and store.
        """
        self.P = {}
        with tqdm(total=self.num_leaves) as pbar:
            for nint in self.leaf_nints:
                self.P[nint] = self.get_leaf_transition_probs(nint)
                pbar.update(1)
        

    def get_paths_from_source(self, costs, source_index, 
                              best_cost, worst_cost, higher_is_better,
                              combine, better, p_n, conditional):
        """
        Use a variant of the Dijkstra algorithm to find the best sequence of transitions from one leaf to all others.
        Where "best" is currently measured by total probability.
        Conditional argument conditions transition probabilities on previous leaf. 
        This makes for sparser data: greater chance of failure but better quality when succeeds.
        """
        costs[source_index][1] = False
        costs[source_index][2] = best_cost
        depth = 0; cond = 'marginal'
        while True:
            depth += 1
            # Sort unvisited leaves by total cost and identify the best one to visit next.
            priority = sorted([c for c in costs if c[4] == False], key=lambda c: c[2], reverse=higher_is_better)
            if priority == []: break # All leaves visited.
            index, previous_index, cost_so_far, _, _ = priority[0]
            # Check if we have reached the end of the accessible leaves.
            if previous_index == None: break 
            # Mark the leaf as visited.
            costs[index][4] = True
            # For conditional transition, condition on previous leaf.
            if conditional and index != source_index: cond = self.leaf_nints[previous_index]
            for next_nint, vals in self.P[self.leaf_nints[index]][p_n][cond].items():
                if next_nint != 0:
                    # Compute cost to this leaf.
                    cost_to_here = combine(cost_so_far, vals[0])
                    # If this is better than the stored one, overwrite.
                    next_index = self.leaf_nints.index(next_nint)
                    if better(cost_to_here, costs[next_index][2]):
                        costs[next_index] = [next_index, index, cost_to_here, vals[0], False]
        # Information to return is previous leaf indices, total costs and one-step costs.
        _, prev, costs_total, costs_one_step, _ = [list(x) for x in zip(*costs)]
        return prev, costs_total, costs_one_step

    
    def compute_paths_matrix(self, reverse=False, cost_by='prob', conditional=False):
        """
        Run the get_paths_from_source method for all leaves and store.
        """
        # Set up some parameters for the search process.
        if cost_by == 'prob':
            best_cost = 1; worst_cost = 0; higher_is_better = True
            combine = lambda a, b: a * b
            better = lambda a, b: a > b
        else: raise Exception(f'cost_by {cost_by} not yet implemented.')
        if reverse: p_n = 'prev'
        else: p_n = 'next'
        print('Computing transition paths matrix...')
        self.path_prev, self.path_costs_total, self.path_costs_one_step = [], [], []
        with tqdm(total=self.num_leaves) as pbar:
            for source_index in range(self.num_leaves):
                # Initialise costs.
                # Elements are [index, index of previous leaf, total cost, immediate cost, visited?]
                # TODO: Pre-populate with places we already know how to get to.
                costs_init = [[i, None, worst_cost, None, False] for i in range(self.num_leaves)]
                # Run the search.
                p, t, o = self.get_paths_from_source(costs_init, source_index, 
                                                     best_cost, worst_cost, higher_is_better,
                                                     combine, better, p_n, conditional)
                self.path_prev.append(p)
                self.path_costs_total.append(t)
                self.path_costs_one_step.append(o)
                pbar.update(1)
        

    def get_leaf_to_leaf_path(self, source, dest): 
        """
        Given a source and destination leaf, get the lowest-cost path between them.
        """
        source_index = self.leaf_nints.index(source)
        dest_index = self.leaf_nints.index(dest)
        prev = self.path_prev[source_index]
        costs_one_step = self.path_costs_one_step[source_index]
        # Reconstruct the path by backtracking.
        index = dest_index; path = [(source, None)]
        while index != source_index:
            path.insert(1, (self.leaf_nints[index], costs_one_step[index]))
            index = prev[index]
            if index == None: return False, False # No path found.
        return path, self.path_costs_total[source_index][dest_index]


    def path_between(self, source=False, source_features={}, source_attributes={}, 
                           dest=False, dest_features={}, dest_attributes={}, 
                           feature_mode = 'contain', try_reuse_df=True):
        """
        Use the get_leaf_to_leaf_path method to find paths between pairs of leaves matching condtions.
        Conditions could be on feature or attribute values.
        Can optionally specify a single start or end leaf.
        """
        if not(try_reuse_df and self.have_df): df = self.to_dataframe()
        df = self.df.loc[self.df['kind']=='leaf'] # Only care about leaves.
        # List source leaf integers.
        if source == False:
            source = self.df_filter(df, source_features, source_attributes, mode=feature_mode).index.values
        elif type(source) == tuple: source = [source]
        # List destination leaf integers.
        if dest == False:
            dest = self.df_filter(df, dest_features, dest_attributes, mode=feature_mode).index.values
        elif type(dest) == tuple: dest = [dest]
        # Find the best path to each leaf matching the condition.
        paths = []
        with tqdm(total=len(source)*len(dest)) as pbar:
            for s in source:
                for d in dest:
                    path, cost = self.get_leaf_to_leaf_path(s, d)
                    pbar.update(1)
                    if path != False:
                        paths.append((path, cost))
        # Sort the paths by their cost and return.
        paths.sort(key=lambda x:x[1], reverse=True)
        return paths


# ===================================================================================================================
# METHODS FOR WORKING WITH COUNTERFACTUAL DATA


    def cf_load_data(self, o, a, r, p, n, regret_steps=np.inf, append=True):
        """
        Counterfactual data looks a lot like target data, 
        but is assumed to originate from a policy other than the target one,
        so must be kept separate.
        """
        assert self.tree != None, 'Must have already grown tree.'
        assert len(o) == len(a) == len(r) == len(p) == len(n), 'All inputs must be the same length.'
        assert min(p) == min(n) >= -1, 'Episode start/end must be denoted by index of -1.'
        if self.cf == None: self.cf = counterfactual()
        # Convert actions into indices.
        if self.classifier: a = np.array([self.action_names.index(c) for c in a]) 
        # Compute return for each new sample.
        g = self.get_returns(r, p, n)
        # Use the extant tree to get a leaf integer for each sample, and predict its value under the target policy.
        R = self.predict(o, attributes=['nint','value'])       
        nints = R['nint']; v_t = R['value']
        # Store the counterfactual data, appending if applicable.
        num_samples_prev = self.cf.num_samples
        if append == False or num_samples_prev == 0:
            self.cf.o, self.cf.a, self.cf.r, self.cf.p, self.cf.n, self.cf.g, self.cf.v_t = o, a, r, p, n, g, v_t
            self.cf.regret = np.empty_like(g) # Empty; compute below.
            self.cf.num_samples = len(o)
        else:
            self.cf.o = np.vstack((self.cf.o, o))
            self.cf.a = np.hstack((self.cf.a, a))
            self.cf.r = np.hstack((self.cf.r, r))
            self.cf.p = np.hstack((self.cf.p, p))
            self.cf.n = np.hstack((self.cf.n, n))
            self.cf.g = np.hstack((self.cf.g, g))
            self.cf.v_t = np.hstack((self.cf.g, v_t))
            self.cf.regret = np.hstack((self.cf.regret, np.empty_like(g))) # Empty; compute below.
            self.cf.num_samples += len(o)
        # Compute regret for each new sample. 
        self.cf_compute_regret(regret_steps, num_samples_prev)
        # Store new samples at nodes by back-propagating.
        samples_per_leaf = {nint:[] for nint in set(nints)}
        for index, nint in zip(np.arange(num_samples_prev, self.cf.num_samples), nints):
            samples_per_leaf[nint].append(index)
        for nint, indices in samples_per_leaf.items():
            address = int_to_bits(nint) 
            self.node(address).cf_indices += indices 
            while address != ():
                ancestor, address = self.parent(address)
                ancestor.cf_indices += indices
        # (Re)compute criticality for all nodes in the tree.
        self.cf_compute_node_criticalities()
        # Finally, use the leaves to estimate criticality for every sample in the training dataset.
        self.c = self.predict(self.o, attributes=['criticality'])


    def cf_compute_regret(self, steps, start_index=0):
        """
        Compute n-step regrets vs the estimated value function
        for samples in the counterfactual dataset,
        optionally specifying a start index to prevent recomputing.
        """
        assert not (start_index > 0 and steps != self.cf.regret_steps), "Can't use different values of regret_steps in an appended dataset; recompute first."
        self.cf.regret[start_index:] = np.nan
        for index in np.argwhere(self.cf.p == -1):
            if index >= start_index:
                ep_indices, regret = self.cf_get_regret_trajectory(index[0], steps)
                self.cf.regret[ep_indices] = regret
        self.cf.regret_steps = steps


    def cf_get_regret_trajectory(self, index, steps=np.inf):
        """
        Compute n-step regrets vs the estimated value function
        for a trajectory of counterfactual samples starting at index.
        """
        # Retrieve all successive samples in the counterfactual episode.
        _, indices = self.sample_episode(self.cf.p, self.cf.n, index)
        o = self.cf.o[indices]
        r = self.cf.r[indices]
        p = self.cf.p[indices]
        n = self.cf.n[indices]
        v_t = self.cf.v_t[indices]
        # Verify steps.
        num_samples = len(r)
        if steps >= num_samples: steps = num_samples-1
        else: assert steps > 0, 'Steps must be None or a positive integer.'

        # Compute n-step returns.
        g = self.get_returns_n_step_ordered_episode(r, p, n, steps)
        # Compute regret = target v - (n-step return + discounted value of sample in n-steps' time).
        v_t_future = np.pad(v_t[steps:], (0, steps), mode='constant') # Pad end of episodes.
        regret = v_t - (g + (v_t_future * (self.gamma ** steps)))
        return indices, regret

        
    def cf_compute_node_criticalities(self):
        """
        The criticality of a node is defined as the mean regret
        of the counterfactual samples lying within it. 
        It can be defined for all nodes, not just leaves.
        """
        assert self.tree != None and self.cf != None and self.cf.num_samples > 0, 'Must have tree and counterfactual data.'
        def recurse(node):
            if node.cf_indices != []: 
                regrets = self.cf.regret[np.array(node.cf_indices)]
                node.criticality_mean = np.nanmean(regrets)
                node.criticality_impurity = np.nanstd(regrets)
            if node.left:
                recurse(node.left)
                recurse(node.right)  
        recurse(self.tree)


    def init_highlights(self, crit_smoothing=5, max_length=np.inf):
        """
        Precompute all the data required for our interpretation 
        of the HIGHLIGHTS algorithm (Amir et al, AAMAS 2018).
        """
        assert self.tree != None and self.cf != None and self.cf.num_samples > 0, 'Must have tree and counterfactual data.'
        smooth_window = (2 * crit_smoothing) + 1
        ep_obs, ep_crit, all_crit, all_timesteps = [], [], [], []
        # Split the training data into episodes and iterate through.
        episodes = self.split_into_episodes(self.p, self.n)
        for ep, indices in enumerate(episodes):
            # Temporally smooth the criticality time series for this episode.
            if crit_smoothing == 0: crit = self.c[indices]
            else: crit = running_mean(self.c[indices], smooth_window)
            ep_crit.append(crit)
            # Store 'unravelled' criticality and timestep indices so can sort globally.
            all_crit += list(crit)
            all_timesteps += [(ep, t) for t in range(len(indices))]            
        # Sort timesteps (across all episodes) by smoothed criticality.
        all_timesteps_sorted = [t for _,t in sorted(zip(all_crit, all_timesteps))]
        # Now iterate through the sorted timesteps and construct highlights in order of criticality.
        self.highlights_keypoint, self.highlights_indices, self.highlights_crit, used = [], [], [], set()
        if max_length != np.inf: max_length = max_length // 2
        with tqdm(total=len(all_timesteps_sorted)) as pbar:
            while all_timesteps_sorted != []:
                # Pop the most critical timestep off all_timesteps_sorted.
                timestep = all_timesteps_sorted.pop()
                pbar.update(1)
                if timestep not in used: # If hasn't been used in a previous highlight.
                    ep, t = timestep
                    # Assemble a list of timesteps either side.
                    trajectory = np.arange(max(0, t-max_length), min(len(episodes[ep]), t+max_length))
                    # Store this highlight and its smoothed criticality time series.
                    self.highlights_keypoint.append(episodes[ep][t])
                    self.highlights_indices.append(episodes[ep][trajectory])
                    self.highlights_crit.append(ep_crit[ep][trajectory])
                    # Record all these timesteps as used.
                    used = used | {(ep, tt) for tt in trajectory}
        self.have_highlights = True 
        
        
    def top_highlights(self, k=1, diversity=0.1, p=2): 
        """
        Assuming init_highlights has already been run, return the k best highlights.
        """
        assert self.have_highlights, 'Must run init_highlights first.'
        assert diversity >= 0 and diversity <= 1, 'Diversity must be in [0,1].'
        if k > 1:
            # Calculate the distance between the opposite corners of the global feature lims box.
            # This is used for normalisation.
            dist_norm = minkowski_distance(self.global_feature_lims[:,0] * self.feature_scales,
                                           self.global_feature_lims[:,1] * self.feature_scales, p=p)
        chosen_highlights_keypoint, chosen_highlights_indices, chosen_highlights_crit, i, n = [], [], [], -1, 0
        while n < k: 
            i += 1
            if i >= len(self.highlights_indices): break
            if diversity > 0 and n > 0:
                # Find the feature-scaled Frechet distance to each existing highlight and normalise.
                # dist_to_others = np.array([scaled_frechet_dist(self.highlights_obs[i],
                #                                                chosen_highlights_obs[j],
                #                                                self.feature_scales, p=p)
                #                            for j in range(n)]) / dist_norm 
                dist_to_others = np.array([minkowski_distance(self.o[self.highlights_keypoint[i]] * self.feature_scales,
                                                              self.o[chosen_highlights_keypoint[j]] * self.feature_scales, p=p)
                                           for j in range(n)]) / dist_norm                
                
                print(i, n, min(dist_to_others), max(dist_to_others))
                
                assert max(dist_to_others) <= 1, 'Error: Scaled-and-normalised distance should always be <= 1!'
                # Ignore this highlight if the smallest Frechet distance is below a threshold.
                if min(dist_to_others) < diversity: continue
            # Store this highlight.
            chosen_highlights_keypoint.append(self.highlights_keypoint[i])
            chosen_highlights_indices.append(self.highlights_indices[i])
            chosen_highlights_crit.append(self.highlights_crit[i])
            n += 1
        return chosen_highlights_indices, chosen_highlights_crit


# ===================================================================================================================
# METHODS FOR TREE DESCRIPTION AND VISUALISATION.


    # TODO: This needs updating.
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


    def to_dataframe(self, out_file=None):
        """
        Represent the nodes of the tree as rows of a Pandas dataframe.
        """
        def recurse(node, partitions=[]):
            # Basic identification.
            data['nint'].append(node.nint)
            data['depth'].append(len(int_to_bits(node.nint)))
            data['kind'].append(('internal' if node.left else 'leaf'))
            # Feature ranges.
            ranges = self.global_feature_lims.copy()
            for f in range(self.num_features):
                for lr, sign in enumerate(('>','<')):
                    thresholds = [p[2] for p in partitions if p[0] == f and p[1] == lr]
                    if len(thresholds) > 0: 
                        # The last partition for each (f, lr) pair is always the most restrictive.
                        ranges[f,lr] = thresholds[-1]
                    data[f'{self.feature_names[f]} {sign}'].append(ranges[f,lr])
            # Population information.
            data['num_samples'].append(node.num_samples)
            data['sample_fraction'].append(node.num_samples / self.num_samples)
            weight_sum = sum(self.w[node.indices])
            data['weight_sum'].append(weight_sum)
            data['weight_fraction'].append(weight_sum / self.w_sum)
            # Volume and density information.
            #   Volume of a leaf = product of feature ranges, scaled by feature_scales_norm.
            volume = np.prod((ranges[:,1] - ranges[:,0]) * feature_scales_norm)
            data['volume'].append(volume)
            data['sample_density'].append(node.num_samples / volume)
            data['weight_density'].append(weight_sum / volume)
            # Action information.
            data['action'].append(node.action_best)
            data['action_impurity'].append(node.action_impurity)
            if self.classifier: 
                data['action_counts'].append(node.action_counts)
                data['weighted_action_counts'].append(node.weighted_action_counts)
            # Value information.
            data['value'].append(node.value_mean)
            data['value_impurity'].append(node.value_impurity)
            # Derivative information.
            data['derivative_impurity'].append(node.d_norm_impurity)
            try: data['transition_impurity'].append(node.transition_impurity)
            except: data['transition_impurity'].append(None)
            # Criticality information.
            data['criticality'].append(node.criticality_mean)
            data['criticality_impurity'].append(node.criticality_impurity)

            # For decision nodes, recurse to children.
            if node.left:
                recurse(node.left, partitions+[(node.feature_index, 1, node.threshold)])
                recurse(node.right, partitions+[(node.feature_index, 0, node.threshold)])

        # Set up dictionary keys.
        #    Basic identification.
        keys = ['nint','depth','kind']
        #    Feature ranges.
        keys += [f'{f} {sign}' for f in self.feature_names for sign in ('>','<')] 
        #    Population information.
        keys += ['num_samples','sample_fraction','weight_sum','weight_fraction','volume','sample_density','weight_density'] 
        #    Action information.
        keys += ['action','action_impurity']
        if self.classifier: keys += ['action_counts','weighted_action_counts']
        #    Value information.
        keys += ['value','value_impurity']
        #    Derivative information.
        keys += ['derivative_impurity']
        keys += ['transition_impurity']
        #    Criticality information.
        keys += ['criticality','criticality_impurity']
        data = {k:[] for k in keys}
        # NOTE: For volume calculations, normalise feature scales by geometric mean.
        # This tends to keep hyperrectangle volumes reasonable.   
        feature_scales_norm = self.feature_scales / np.exp(np.mean(np.log(self.feature_scales)))
        # Populate dictionary by recursion through the tree.
        recurse(self.tree)        
        # Convert into dataframe.
        self.df = pd.DataFrame.from_dict(data).set_index('nint')
        self.have_df = True
        # If no out file specified, just return.
        if out_file == None: return self.df
        else: self.df.to_csv(out_file+'.csv', index=False)


    def df_filter(self, df, features={}, attributes={}, mode='overlap'):
            """
            Filter the subset of nodes that overlap with (or are entirely contained within)
            a set of feature ranges, and / or have attributes within specified ranges.
            If using features, compute the proportion of overlap for each.
            """
            # Build query.
            query = []; feature_ranges = []
            for f, r in features.items():
                feature_ranges.append(r)
                # Filter by features.
                if mode == 'overlap':
                    # Determine whether two ranges overlap:
                    # https://stackoverflow.com/questions/325933/determine-whether-two-date-ranges-overlap/325964#325964
                    query.append(f'`{f} <`>={r[0]}')
                    query.append(f'`{f} >`<={r[1]}')
                elif mode == 'contain':
                    query.append(f'`{f} >`>={r[0]}')
                    query.append(f'`{f} <`<={r[1]}')
            for attr, r in attributes.items():
                # Filter by attributes.
                query.append(f'{attr}>={r[0]} & {attr}<={r[1]}')
            # Filter dataframe.
            df = df.query(' & '.join(query))
            if features != {}:
                # If using features, compute overlap proportions,
                # and store this in a new column of the dataframe.
                # There's a lot of NumPy wizardry going on here!
                feature_ranges = np.array(feature_ranges)
                node_ranges = np.dstack((df[[f'{f} >' for f in features]].values,
                                    df[[f'{f} <' for f in features]].values))
                overlap = np.maximum(0, np.minimum(node_ranges[:,:,1], feature_ranges[:,1]) 
                                    - np.maximum(node_ranges[:,:,0], feature_ranges[:,0]))
                df['overlap'] = np.prod(overlap / (node_ranges[:,:,1] - node_ranges[:,:,0]), axis=1)                         
            return df

    
    def treemap(self, features, attributes=[None], lims=[], 
                  visualise=True, axes=[],
                  action_colours=None, cmap_percentiles=[5,95], cmap_midpoints=[], density_percentile=90,
                  alpha_by_density=False, edge_colour=None, show_nints=False, try_reuse_df=True):
        """
        Create a treemap visualisation across one or two features, 
        possibly projecting across all others.
        """
        if type(features) in (str, int): features = [features]
        n_f = len(features)
        assert n_f in (1,2), 'Can only plot in 1 or 2 dimensions.'
        if not(try_reuse_df and self.have_df): df = self.to_dataframe()
        df = self.df.loc[self.df['kind']=='leaf'] # Only care about leaves.
        if lims == []: lims = [[None,None]] * n_f
        # For any lim that = None, replace with the global min or max.
        lims = np.array(lims).astype(float)
        fi = [self.feature_names.index(f) for f in features]
        np.copyto(lims, self.global_feature_lims[fi], where=np.isnan(lims))
        cmaps = {'action':custom_cmap, # For regression only.
                 'action_impurity':custom_cmap.reversed(),
                 'value':custom_cmap,
                 'value_impurity':custom_cmap.reversed(),
                 'derivative_impurity':custom_cmap.reversed(),
                 'transition_impurity':custom_cmap.reversed(),
                 'criticality':custom_cmap,
                 'criticality_impurity':custom_cmap.reversed(),
                 'sample_density':matplotlib.cm.gray,
                 'weight_density':matplotlib.cm.gray,
                 None:None
                 }
        if type(attributes) == str or attributes == None: attributes = [attributes]
        n_a = len(attributes)
        for attr in attributes: 
            assert attr in cmaps, 'Invalid attribute.'
        # If doing alpha_by_density, need to evaluate sample_density even if not requested.
        attributes_plus = attributes.copy()
        if alpha_by_density and not 'sample_density' in attributes: 
            attributes_plus += ['sample_density']
        if cmap_midpoints == []: 
            # If cmap midpoints not specified, use defaults.
            cmap_midpoints = [None for f in attributes]   
        if n_f < self.num_features: marginalise = True
        else: marginalise = False
        regions = {}  
        if not (attributes == [None] and edge_colour == None):                
            if not marginalise:
                # This is easy: can just use leaves directly.
                if n_f == 1: height = 1
                for nint, leaf in df.iterrows():
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
                    if n_f == 1: xy.append(0)
                    m = nint
                    regions[m] = {'xy':xy, 'width':width, 'height':height, 'alpha':1}  
                    for attr in attributes_plus: 
                        if attr != None: regions[m][attr] = leaf[attr]   
            else:
                # Get all unique values mentioned in partitions for these features.
                f1 = features[0]
                p1 = np.unique(df[[f1+' >',f1+' <']].values) # Sorted by default.
                r1 = np.vstack((p1[:-1],p1[1:])).T # Ranges.
                if n_f == 2: 
                    f2 = features[1]
                    p2 = np.unique(df[[f2+' >',f2+' <']].values) 
                    r2 = np.vstack((p2[:-1],p2[1:])).T
                else: r2 = [[None,None]]
                for m, ((min1, max1), (min2, max2)) in enumerate(tqdm(np.array([[i,j] for i in r1 for j in r2]))):
                    if min1 >= lims[0][1] or max1 <= lims[0][0]: continue # Ignore if leaf is outside lims.
                    min1 = max(min1, lims[0][0])
                    max1 = min(max1, lims[0][1])
                    width = max1 - min1
                    feature_ranges = {features[0]: [min1, max1]}
                    if n_f == 1: 
                        xy = [min1, 0]
                        height = 1
                    else: 
                        if min2 >= lims[1][1] or max2 <= lims[1][0]: continue
                        min2 = max(min2, lims[1][0])
                        max2 = min(max2, lims[1][1])
                        feature_ranges[features[1]] = [min2, max2]
                        xy = [min1, min2]
                        height = max2 - min2   
                    regions[m] = {'xy':xy, 'width':width, 'height':height, 'alpha':1}  
                    if attributes != [None]:           
                        # Find "core": the leaves that overlap with the feature range(s).
                        core = self.df_filter(df, features=feature_ranges)
                        for attr in attributes_plus:
                            if attr == None: pass
                            elif attr == 'action' and self.classifier:
                                # Special case for action with classification: discrete values.
                                regions[m][attr] = np.argmax(np.dot(np.vstack(core['weighted_action_counts'].values).T, 
                                                                            core['overlap'].values.reshape(-1,1)))
                            else: 
                                if attr in ('sample_density','weight_density'): normaliser = 'volume' # Another special case for densities.
                                else:                                           normaliser = 'weight_sum'
                                # Take contribution-weighted mean across the core.
                                norm_sum = np.dot(core[normaliser].values, core['overlap'].values)
                                # NOTE: Averaging process assumes uniform data distribution within leaves.
                                core['contrib'] = core.apply(lambda row: (row[normaliser] * row['overlap']) / norm_sum, axis=1)
                                regions[m][attr] = np.nansum(core[attr].values * core['contrib'].values)          
        if visualise:
            if n_f == 1:
                if axes != []: ax = axes
                else: _, ax = matplotlib.pyplot.subplots(); axes = ax
                ax.set_xlabel(features[0]); ax.set_xlim(lims[0])
                ax.set_yticks(np.arange(n_a)+0.5)
                ax.set_yticklabels(attributes)
                ax.set_ylim([0,max(1,n_a)])
                if alpha_by_density:
                    # If doing alpha_by_density, add background.
                    ax.add_patch(matplotlib.patches.Rectangle(xy=[lims[0][0],0], width=lims[0][1]-lims[0][0], height=n_a, 
                                            facecolor='k', hatch=None, edgecolor=None, zorder=-11))
            else:
                offset = np.array([0,0])
            # If doing alpha_by_density, precompute alpha values.
            if alpha_by_density:
                density_list = [r['sample_density'] for r in regions.values()]
                amin = np.nanmin(density_list)
                amax = np.nanpercentile(density_list, density_percentile) 
                alpha_norm = matplotlib.colors.LogNorm(vmin=amin, vmax=amax)
                for m, region in regions.items():
                    regions[m]['alpha'] = min(alpha_norm(region['sample_density']), 1)
            for a, attr in enumerate(attributes):
                if n_f == 1:
                    offset = np.array([0,a])
                else:
                    if len(axes) <= a: 
                        axes.append(matplotlib.pyplot.subplots()[1])
                    ax = axes[a]
                    ax.set_title(attr)
                    ax.set_xlabel(features[0]); ax.set_xlim(lims[0])
                    ax.set_ylabel(features[1]); ax.set_ylim(lims[1])    
                    if alpha_by_density:
                        # If doing alpha_by_density, add background.
                        ax.add_patch(matplotlib.patches.Rectangle(xy=[lims[0][0],lims[1][0]], width=lims[0][1]-lims[0][0], height=lims[1][1]-lims[1][0], 
                                    facecolor='k', hatch=None, edgecolor=None, zorder=-11))
                if attr == None: pass
                elif attr == 'action' and self.classifier:
                    assert action_colours != None, 'Specify colours for discrete actions.'
                else: 
                    attr_list = [r[attr] for r in regions.values()]
                    if attr in ('sample_density','weight_density'):
                        # For density attributes, use a logarithmic cmap and clip at a specified percentile.
                        vmin = np.nanmin(attr_list)
                        vmax = np.nanpercentile(attr_list, density_percentile) 
                        colour_norm = matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax)
                    else: 
                        vmin = np.nanpercentile(attr_list, cmap_percentiles[0])
                        vmax = np.nanpercentile(attr_list, cmap_percentiles[1])
                        vmid = cmap_midpoints[a]
                        if vmid != None: 
                            # Make symmetric.
                            half_range = max(vmax-vmid, vmid-vmin)
                            vmin = vmid - half_range; vmax = vmid + half_range
                        colour_norm = MidpointNormalize(vmin=vmin, vmax=vmax, midpoint=vmid)
                    dummy = ax.imshow(np.array([[vmin,vmax]]), aspect='auto', cmap=cmaps[attr], norm=colour_norm)
                    dummy.set_visible(False)
                    if n_f == 1:
                        axins = inset_axes(ax,
                                           width='3%',  
                                           height=f'{(100/n_a)-1}%',  
                                           loc='lower left',
                                           bbox_to_anchor=(1.01, a/n_a, 1, 1),
                                           bbox_transform=ax.transAxes,
                                           borderpad=0,
                                           )
                        matplotlib.pyplot.colorbar(dummy, cax=axins)
                    else: matplotlib.pyplot.colorbar(dummy, ax=ax)
                for m, region in regions.items():
                    if attr == None: colour = 'w'
                    elif attr == 'action' and self.classifier: colour = action_colours[region[attr]]
                    else: colour = cmaps[attr](colour_norm(region[attr]))
                    # Don't apply alpha to density plots themselves.
                    if attr in ('sample_density','weight_density'): alpha = 1
                    else: alpha = region['alpha']
                    ax.add_patch(matplotlib.patches.Rectangle(
                                 xy=region['xy']+offset, width=region['width'], height=region['height'], 
                                 facecolor=colour, edgecolor=edge_colour, alpha=alpha, zorder=-10))
                    # Add leaf integer.
                    if not marginalise and show_nints: 
                        ax.text(region['xy'][0]+region['width']/2, region['xy'][1]+region['height']/2, m, 
                                horizontalalignment='center', verticalalignment='center')
            # Some quick margin adjustments.
            matplotlib.pyplot.tight_layout()
            if n_f == 1: matplotlib.pyplot.subplots_adjust(right=0.85)
            elif n_a == 1: axes = axes[0]
        return regions, axes


    def plot_transitions_2D(self, path, features, tightening=1, tighten_thresh=1e-4,
                            ax=None, colour='k', alpha=1, try_reuse_df=True):
        """
        Given a sequence of transitions between leaves, plot on the specified feature axes.
        Optionally do iterative averaging to tighten the path rather than using leaf centroids.
        """
        if not(try_reuse_df and self.have_df): self.to_dataframe()
        assert tightening >= 0 and tightening <= 1
        n_f = len(features)
        # List the columns to query from the dataframe.
        cols = []
        for f in features: cols += [f+' >']
        for f in features: cols += [f+' <']
        # Create a matrix containing the hyperrectangle boundaries for the leaves visited.
        # This has shape 2 * len(path) * len(features).
        lims = np.stack(np.split(self.df.loc[[p[0] for p in path]][cols].values, 2, axis=1))
        # Initialise the points to plot as the centroids of the hyperrectangles.
        centroids = np.mean(lims, axis=0)
        pts = centroids.copy()
        # If doing tightening, iteratively average each point.
        if tightening > 0:
            i = 0
            scales = np.array([self.feature_scales[self.feature_names.index(f)] for f in features])
            while True:
                pts_new = pts.copy()
                for f in range(n_f):
                    pts_new[:,f] = np.convolve(pts_new[:,f],[0.5,0,0.5], 'full')[1:-1]
                pts_new = np.clip(pts_new, lims[0], lims[1])
                delta = np.max((pts_new - pts) * scales)
                if i == 0: 
                    if delta == 0: delta = 1 # Prevents div/0 error.
                    delta_0 = delta
                    
                elif delta / delta_0 < tighten_thresh: break
                pts = pts_new; i += 1
            # Take the weighted average of the relaxed and centroid points.
            pts = (pts * tightening) + (centroids * (1 - tightening))
        return self.plot_trajectory_2D(pts, ax=ax, colour=colour, alpha=alpha)


    def plot_trajectory_2D(self, pts, features=[], ax=None, colour='k', alpha=1):
        """
        Plot a trajectory of observations in a two-feature plane.
        """
        pts = np.array(pts)
        n_f = pts.shape[1]
        if n_f != 2:
            assert n_f == self.num_features, 'Data shape must match num_features.'
            assert len(features) == 2, 'Need to specify two features for projection.'
            # This allows plotting of a projected path onto two specified features.
            pts = pts[:,[self.feature_names.index(f) for f in features]]   
        if ax == None: _, ax = matplotlib.pyplot.subplots()
        ax.plot(pts[:,0], pts[:,1], c=colour, alpha=alpha)
        ax.scatter(pts[0,0], pts[0,1], c='y', alpha=alpha, zorder=10)
        ax.scatter(pts[-1,0], pts[-1,1], c='k', alpha=alpha, zorder=10)
        return pts, ax


    def cf_scatter_regret(self, features, indices=None, lims=[], ax=None):
        """
        Create a scatter plot showing all counterfactual samples,
        coloured by their regret.
        """
        if not ax: _, ax = matplotlib.pyplot.subplots()
        assert len(features) == 2, 'Can only plot in 1 or 2 dimensions.'
        # If indices not specified, use all.
        if indices == None: indices = np.arange(len(self.cf.o))
        indices = indices[~np.isnan(self.cf.regret[indices])] # Ignore NaNs.
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
        matplotlib.pyplot.colorbar(dummy, ax=ax, orientation='horizontal') 
        colours = matplotlib.matplotlib.cm.Reds((regret - lower_perc) / perc_range)
        # Plot.
        ax.scatter(o[:,0], o[:,1], s=0.5, color=colours)
        return ax


    def plot_leaf_derivatives_2D(self, features=[], ax=None, 
                                 colour='k', lengthscale=1, linewidth=0.005, try_reuse_df=True):
        """
        xxx
        """
        if not(try_reuse_df and self.have_df): self.to_dataframe()
        if len(features) != 2: 
            assert self.num_features == 2; features = self.feature_names
        # List the columns to query from the dataframe.
        cols = []
        for f in features: cols += [f+' >']
        for f in features: cols += [f+' <']
        lims = np.stack(np.split(self.df.loc[self.leaf_nints][cols].values, 2, axis=1))
        centroids = np.mean(lims, axis=0) # Position arrows at leaf centroids.
        # Get mean derivative for each leaf.
        d_mean = np.array([self.node(nint).derivative_mean for nint in self.leaf_nints])
        # Plot arrows.
        if ax == None: _, ax = matplotlib.pyplot.subplots()
        matplotlib.pyplot.quiver(centroids[:,0], centroids[:,1], d_mean[:,0], d_mean[:,1], 
                                 pivot='mid', angles='xy', scale_units='xy', units='xy', 
                                 color=colour, scale=1/lengthscale, width=linewidth, minshaft=1)
        return ax


# ===================================================================================================================
# NODE CLASS.


class Node():
    def __init__(self, 
                 nint, 
                 num_samples, 
                 indices, 
                 ):
        # These are the basic attributes; more are added elsewhere.
        self.nint = nint
        self.indices = indices
        self.num_samples = num_samples
        self.left = None
        self.right = None

    
# ===================================================================================================================
# CLASS FOR HOLDING EXPLORATORY DATA.


class counterfactual(): 
    def __init__(self): 
        self.num_samples = 0 # Initially dataset is empty.
        self.regret_steps = None


# ===================================================================================================================
# SOME EXTRA BITS.

# Colour normaliser.
# From https://github.com/mwaskom/seaborn/issues/1309#issue-267483557
class MidpointNormalize(matplotlib.colors.Normalize):
    def __init__(self, vmin, vmax, midpoint=None, clip=False):
        #if vmin == vmax: self.degenerate = True
        #else: self.degenerate = False
        if midpoint == None: self.midpoint = (vmax + vmin) / 2
        else: self.midpoint = midpoint
        matplotlib.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        #if self.degenerate: return 'w'
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

# A custom colour map.
cdict = {'red':   [[0.0,  1.0, 1.0],
                   [0.5,  0.25, 0.25],
                   #[0.5, 0.6, 0.6],
                   [1.0,  0.0, 0.0]],
         'green': [[0.0,  0.0, 0.0],
                   [0.5,  0.25, 0.25],
                   #[0.5, 0.4, 0.4],
                   [1.0,  0.8, 0.8]],
         'blue':  [[0.0,  0.0, 0.0],
                   [0.5,  1.0, 1.0],
                   #[0.5, 0.0, 0.0],
                   [1.0,  0.0, 0.0]]}                
custom_cmap = matplotlib.colors.LinearSegmentedColormap('custom_cmap', segmentdata=cdict)

# Fast way to calculate running mean.
# From https://stackoverflow.com/a/43200476
import scipy.ndimage.filters as ndif
def running_mean(x, N):
    x = np.pad(x, N // 2, mode='constant', constant_values=(x[0],x[-1]))
    return ndif.uniform_filter1d(x, N, mode='constant', origin=-(N//2))[:-(N-1)]

# Convert sequence of bits (e.g. leaf address) to an integer.
# NOTE: Need to prepend a "1" to disambiguate between 00000 and 000.
def bits_to_int(bits):
    out = 0
    for bit in [1] + list(bits): out = (out << 1) | bit
    return out

# Convert the other way.
def int_to_bits(integer): return tuple(int(i) for i in bin(integer)[3:])

# Frechet distance computation with scaled dimensions.
# Slightly changed from https://github.com/cjekel/similarity_measures/blob/master/similaritymeasures/similaritymeasures.py.
def scaled_frechet_dist(X, Y, scales, p):    
    X = X * scales
    Y = Y * scales
    n, m = len(X), len(Y)
    ca = np.multiply(np.ones((n, m)), -1)
    ca[0, 0] = minkowski_distance(X[0], Y[0], p=p)
    for i in range(1, n):
        ca[i, 0] = max(ca[i-1, 0], minkowski_distance(X[i], Y[0], p=p))
    for j in range(1, m):
        ca[0, j] = max(ca[0, j-1], minkowski_distance(X[0], Y[j], p=p))
    for i in range(1, n):
        for j in range(1, m):
            ca[i, j] = max(min(ca[i-1, j], ca[i, j-1], ca[i-1, j-1]),
                           minkowski_distance(X[i], Y[j], p=p))
    return ca[n-1, m-1]