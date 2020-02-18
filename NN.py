import numpy as np
import dateutil.parser as dup

class model:

	def __init__(self, in_size, hidden_layers, out_size, weights = ['norm',1/np.sqrt(3)], hidden_activation=['logistic'], output_activation=[None]):

		self.layers = [in_size] + hidden_layers + [out_size]
		
		# If weights not provided, initialise them randomly with a nan on the 0th layer (makes indexing easier).
		if weights[0] == 'norm':
			self.weights = [np.nan]
			for layer in range(1,len(self.layers)):
				# Append a randomised numpy array (variance 1/3) of dims (len(layer l) * (len(layer l - 1) + 1 {for bias}).
				self.weights.append(np.random.normal(0, weights[1], [self.layers[layer],(self.layers[layer - 1] + 1)]))

		# **ADD SHAPE VALIDATION
		else: self.weights = weights

		# Store activation methods.
		self.hidden_activation = hidden_activation
		self.output_activation = output_activation

		# Store input/output size and layer topology.
		self.in_size = in_size; self.out_size = out_size

	def __str__(self):
		return 'Topology = '+str(self.layers)+'; hidden activation = '+str(self.hidden_activation)+'; output activation = '+str(self.output_activation)

	def activate(self, z, a): 
		if a[0] == 'logistic': return 1 / (1 + np.exp(-z))
		elif a[0] == 'tanh': return np.tanh(z)
		elif a[0] == 'relu': return np.maximum(z, 0.)
		elif a[0] == 'leaky-relu': return np.array([x if x > 0 else x * a[1] for x in z])
		# elif a[0] == 'softmax': return
		elif a[0] == None: return z

	def activate_diff(self, z, a):
		if a[0] == 'logistic': return self.activate(z, a) * (1 - self.activate(z, a))
		elif a[0] == 'tanh': return 1 - (self.activate(z, a)**2)
		elif a[0] == 'relu': return (z > 0).astype(float)
		elif a[0] == 'leaky-relu': return np.array([1. if x > 0 else a[1] for x in z])
		elif a[0] == None: return np.ones_like(z) 
		else: raise Exception('Diff for '+a[0]+' not yet implemented!') 

	def predict(self, features, include_internals = False):
		if len(features) != self.in_size: raise Exception('Feature vector '+str(features)+' has wrong dims for network!'); return
		z = [np.nan]; a = [list(features) + [1]]
		for layer in range(1,len(self.layers)-1):
			z.append(np.sum(a[-1] * self.weights[layer],axis=1))
			a.append(np.append(self.activate(z[-1], self.hidden_activation),1))
		z.append(np.sum(a[-1] * self.weights[layer+1],axis=1))
		y = self.activate(z[-1], self.output_activation)

		if self.out_size == 1: y = y[0]
		if include_internals: return y, z, a
		else: return y

	def batch_predict(self, batch, include_internals = False):
		results = []
		for x in batch: 
			a = self.predict(x, include_internals)
			results.append(a)
		return results

	def get_grads(self, z, a, error, include_in_grad = False):

		# **CLEAN UP ALL THE RESHAPING STUFF**

		if type(error) == np.float64 or type(error) == float: error = [error] # Convert to list if single output neuron.
		dJ_by_da = np.array(error).reshape(-1,1)

		grads = [np.array([]) for l in range(len(self.layers))]
		for layer in reversed(range(1,len(self.layers))):

			# Differential of activation function.
			if layer == len(self.layers)-1: da_by_dz = self.activate_diff(z[layer], self.output_activation)
			else: da_by_dz = self.activate_diff(z[layer], self.hidden_activation) 

			# Update weights in this layer.
			dJ_by_dz = da_by_dz.reshape(-1,1) * dJ_by_da
			grads[layer] = dJ_by_dz * a[layer-1]

			# Then compute desired changes in activations of previous layer (excluding bias term).
			if layer > 1 or include_in_grad: 
				dJ_by_da = np.dot(np.transpose([x[:-1] for x in self.weights[layer]]), dJ_by_dz)

		if include_in_grad: return grads, dJ_by_da
		else: return grads 

	def online_learn(self, features, rate, error = [None], target = None, error_type = None):

		y, z, a = self.predict(features, include_internals = True)

		if error[0] == None:
			if error_type == None: error = target - y
			elif error_type == 'log': 
				if target == 1: error = np.maximum(np.log(y), -5.)
				elif target == 0: error = np.minimum(-np.log(1-y), 5.)
				else: raise Exception('Log error requires a target of 0 or 1 (classification)!') 

			#print(y, target, error)

		self.weights = [ self.weights[i] - grads*rate for i,grads in enumerate(self.get_grads(z, a, error))]

	#def batch_learn(self, features_batch, )
	#
	#
	#
	#
	#