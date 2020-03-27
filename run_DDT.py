from my.DDT import DifferentiableDecisionTree

import numpy as np
import matplotlib.pyplot as plt
from joblib import dump
    
input_size = 1
output_size = 1
num_instances = 100   
depth = 3

dt = DifferentiableDecisionTree(depth=depth, input_size=input_size, output_size=output_size, lr_y=1, lr_w=1)

_, ax = plt.subplots(); plt.ion(); plt.show()

X = np.random.rand() + np.sort(np.random.rand(num_instances, input_size), axis=0) #np.random.rand() 
T = np.sin(16*X)+ 0.5*np.random.rand(np.shape(X)[0],np.shape(X)[1])

ax.scatter(X, T, color='k', s=1)

for i in range(20000):

    #idx = np.random.randint(len(X), size=64)

    dt.train_on_batch(X,T)#X[idx,:], T[idx,:])
    Y = dt.predict(X)
    ax.plot(X, Y, color='g')
    mean_E = np.abs(T - Y).mean()
    print(i, mean_E)
    plt.pause(0.0001)
    l = ax.lines.pop(-1); del l

dump(dt, 'sin(16x).joblib')

plt.ioff(); plt.show()