from my.DDT import DifferentiableDecisionTree

import numpy as np
import matplotlib.pyplot as plt
from joblib import dump
    

num_instances = 500   
depth = 3



#X = -np.random.rand() + np.sort(np.random.rand(num_instances, input_size), axis=0) #np.random.rand() 
#T = X**2 + 0.1*np.random.rand(np.shape(X)[0],np.shape(X)[1])

# 2 dimensional.
X = np.array([np.random.uniform([-5,-5],[5,5]) for _ in range(num_instances)])
#T = np.sin(X[:,0] + X[:,1]).reshape(-1,1)
T = np.sqrt(X[:,0]**2 + X[:,1]**2).reshape(-1,1)

dt = DifferentiableDecisionTree(depth=depth, input_size=np.shape(X)[1], output_size=np.shape(T)[1], lr_y=0.5, lr_w=0.5)

# #dt.initialise_leaves(T)

# _, ax = plt.subplots(); plt.ion(); plt.show()
# ax.scatter(X, T, color='k', s=1)

for i in range(5000):

    dt.train_on_batch(X,T)
    Y, mus, _ = dt.predict(X, composition=True)
    mean_E = np.abs(T - Y).mean()
    print(i, mean_E)
    #print([leaf.y for leaf in dt.leaves])
    #print(mus[0], Y[0])
    #print(mus[-1], Y[-1])
    #ax.plot(X, Y, color='g')
    #plt.pause(0.0001)
    #l = ax.lines.pop(-1); del l

dump(dt, 'sqrt(X[0]**2 + X[1]**2).joblib')

# plt.ioff(); plt.show()