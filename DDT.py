
"""
Implementation of a differentiable decision tree regression model.
Originally introducted in:    Suarez, A., and J.F. Lutsko. "Globally Optimal Fuzzy Decision Trees for Classification and Regression."
                              IEEE Transactions on Pattern Analysis and Machine Intelligence 21, no. 12 (December 1999): 1297–1311.

Some details also taken from: Silva, A., et al. "Optimization Methods for Interpretable Differentiable Decision Trees in Reinforcement Learning". 
                              ArXiv:1903.09338 [Cs, Stat], 21 February 2020. http://arxiv.org/abs/1903.09338.

                              Frosst, N., and G. Hinton. "Distilling a Neural Network Into a Soft Decision Tree". 
                              ArXiv:1711.09784 [Cs, Stat], 27 November 2017. http://arxiv.org/abs/1711.09784.


TODO: Initialise beta using smoothness of data (Fourier transform?
      Initialise y based on density in data - just randomly sample y values from data itself?
        Risk then of converging to one if highly unbalanced.



"""

import numpy as np


class DifferentiableDecisionTree:
    def __init__(self,
                 depth,
                 input_size,   
                 output_size,
                 lr_y,
                 lr_w,
                 y_lims,       # Used to initialise and place limits on per_leaf predictions.

                 beta = 1,     # Temperature parameter moderating the degree of crispness in the tree. Higher value = more crisp.
                 w_init_sd = 0 # Standard deviation for initialising weights.
                ):

        self.depth = depth
        self.num_leaves = 2**self.depth
        self.input_size = input_size
        self.output_size = output_size
        assert len(y_lims) == 2 and len(y_lims[0]) == output_size, 'Need upper and lower lim for each output.'
        self.y_lims = y_lims
        self.beta = beta                
        self.w_init_sd = w_init_sd
        self.lr_y = lr_y
        self.lr_w = lr_w

        self._grow_tree()

    
    def _grow_tree(self):
        """Initialise tree structure."""
        def recurse(depth_remaining):
            node = Node()
            if depth_remaining == 0: 
                node._init_leaf(self.y_lims)
                self.leaf_nodes.append(node)
            else:
                node._init_internal(self.input_size, sd_w=self.w_init_sd) 
                node.left = recurse(depth_remaining-1)
                node.right = recurse(depth_remaining-1)
                self.internal_nodes.append(node)
            return node
        self.leaf_nodes = []
        self.internal_nodes = []
        self.tree = recurse(self.depth)
        self.w = np.array([node.w for node in self.internal_nodes])
        self.y = np.array([node.y for node in self.leaf_nodes])
        # Internal ancestors of each leaf node, along with the decision direction.
        offset = np.cumsum([0] + [2**d for d in range(self.depth-1)])
        self.ancestors = []
        for l in range(2**self.depth):
            path = self._depth_node_to_path(self.depth, l)
            ancestors = np.array([0]+[int(path[:d], 2) for d in range(1,self.depth)]) + offset
            signs = np.array([int(s) for s in path])
            self.ancestors.append([ancestors, signs])

    
    # def initialise_weights_from_average(self, x):
    #     """Initialise weights using a single 'average' instance x.
    #     Set weights so that its membership is split evenly across the leaves.
    #     """
    #     x = np.append(1, x)
    #     print(self.tree._mu(x, self.beta))

    
    def update_step(self, X, T, print_mae_before=False):
        """Perform  a gradient update using X and T."""
        # Reshape input if required.
        X = np.array(X); T = np.array(T)
        #if len(np.shape(X)) == 1: X = X.reshape(1,-1); T = T.reshape(1,-1)
        num_instances = len(X)
        # Add column of 1s to X for bias term.
        X = np.c_[ np.ones(num_instances), X ]
        # Compute loss.
        Y, mus, ys = self.predict(X, have_preprocessed=True, composition=True)
        L = T - Y

        if print_mae_before: print(f'MAE before = {np.mean(np.abs(L))}')
        
        # Update leaf parameters.
        dL_dys = np.mean(mus * L, axis=0)
        for l, node in enumerate(self.leaf_nodes):
            node.y += self.lr_y * dL_dys[l]
            # Clip within limits.
            if self.y_lims != None:
                node.y = np.clip(node.y, self.y_lims[0], self.y_lims[1])

        # Update internal node parameters.
        mus_one_level_down = mus
        ys_one_level_down = np.repeat(ys[None,:,:], num_instances, axis=0)
        for d in reversed(range(self.depth)):
            mus_this_level = np.empty((num_instances, 2**d))
            ys_this_level = np.empty((num_instances, 2**d, self.output_size))
            for n in range(2**d):
                # Collect membership and partial predictions from children.
                mu_children = mus_one_level_down[:,2*n:2*n+2]
                y_children = ys_one_level_down[:,2*n:2*n+2,:]
                mus_this_level[:,n] = np.sum(mu_children, axis=1)
                mu_children_norm = mu_children / mus_this_level[:,n][:,None]
                ys_this_level[:,n,:] = np.einsum('ij,ijk->ik', mu_children_norm, y_children)
                self._update_node_weights(d, n, X, L, mus_this_level[:,n], mu_children_norm, y_children)
            # Store for next level.
            mus_one_level_down = mus_this_level
            ys_one_level_down = ys_this_level
        
        # Update w and y arrays to reflect per-node changes.
        self.w = np.array([node.w for node in self.internal_nodes])
        self.y = np.array([node.y for node in self.leaf_nodes])

    
    def predict(self, X, have_preprocessed=False, composition=False):
        """Predict for a dataset X, optionally returning per-leaf membership and prediction."""
        # Reshape input if required.
        if not have_preprocessed:
            X = np.array(X)
            if len(X.shape) == 1: X = X.reshape(1,-1) 
            # Add column of 1s to X for bias term.
            X = np.c_[ np.ones(X.shape[0]), X ]

        # Get mu values from all internal nodes.
        internal_mus = (1 / ( 1 + np.exp( self.beta * np.inner( self.w, X ) )))[None,:,:]
        internal_mus = np.insert(internal_mus, 1, 1-internal_mus[0],axis=0) # Compute 1 minus these values.

        # Multiply these together to get leaf mus.
        mus = np.zeros((X.shape[0], 2**self.depth))
        for l in range(2**self.depth): 
            ancestors, signs = self.ancestors[l]
            mus[:,l] = np.prod(internal_mus[signs, ancestors], axis=0)      
        Y = np.dot(mus, self.y)
        if composition: return Y, mus, self.y
        return Y

    
    # def _propagate(self, x, node, mu=1):
    #     """Propagate a single instance x through the tree starting at node,
    #     and collect per-leaf membership and prediction."""
    #     if node.isleaf: return np.array([mu]), np.array(node.y)
    #     mu_left = node._mu(x, self.beta)
    #     subtree_left, y_left = self._propagate(x, node.left, mu*mu_left)
    #     subtree_right, y_right = self._propagate(x, node.right, mu*(1-mu_left))
    #     return np.append(subtree_left, subtree_right), np.vstack((y_left, y_right))


    def _update_node_weights(self, d, n, X, L, mu, mu_children_norm, y_children):
        """Update the weight parameters for a single decision node."""
        # Navigate to the node.
        node = self.tree
        for l_or_r in self._depth_node_to_path(d, n):
            if l_or_r == '0': node = node.left
            if l_or_r == '1': node = node.right

        # Compute gradient of loss with respect to the membership of the left child.
        dL_dmu = np.mean((L * mu[:,None]) * (y_children[:,0,:] - y_children[:,1,:]), axis=1) # Take mean across output dimensions.

        # Compute gradient of left-child membership with respect to the parameters w.
        dmu_dw = -X * self.beta * (mu_children_norm[:,0] * mu_children_norm[:,1])[:,None] 

        # Apply weight updates, taking mean across instances.
        node.w += self.lr_w * np.mean(dL_dmu[:,None] * dmu_dw, axis=0)


    def _depth_node_to_path(self, d, n):
        """Given a (depth, node) pair, return the path-from-root as a string of 0s and 1s, where 0 = left and 1 = right."""
        if d == 0: return ''
        return str(bin(n))[2:].zfill(d)


class Node:
    def __init__(self):
        self.isleaf = False

    
    def _init_leaf(self, y_lims):
        """Use random uniform sample to initialise predictions for a leaf node."""
        self.isleaf = True
        self.y = np.random.uniform(low=y_lims[0], high=y_lims[1]) 

    
    def _init_internal(self, input_size, sd_w):
        """Use Gaussian to initialise weights for an internal node."""
        self.w = np.random.normal(scale=[sd_w*input_size]+[sd_w]*input_size, size=input_size+1)

    
    # def _mu(self, X,  beta):
    #     #print(beta)
    #     """Membership function of this node's left child for an input X."""
    #     return 1 / ( 1 + np.exp( beta * np.inner( self.w, X ) ))