from my.DDT import DifferentiableDecisionTree

import numpy as np
import matplotlib.pyplot as plt
from joblib import load

ddt = load('sqrt(X[0]**2 + X[1]**2).joblib')

#Y, mus, ys = ddt.predict([1], composition=True)

"""
Seems to be producing leaves whose values are way outside the distribution of the data, 
but 'cancel each other out' - this isn't a great way of solving the problem, and makes it hard to make crisp.
"""

fig = plt.figure()
ax = fig.gca(projection='3d')

# Make data.
X = np.arange(-5, 5, 0.25)
Y = np.arange(-5, 5, 0.25)
X, Y = np.meshgrid(X, Y)
Z = np.zeros_like(X)

for i in range(np.shape(X)[0]):
    for j in range(np.shape(X)[1]):
        Z[i,j] = ddt.predict([X[i,j],Y[i,j]])[0,0]

ax.plot_surface(X, Y, Z)

fig = plt.figure()
ax = fig.gca(projection='3d')

# Make data.
X = np.arange(-5, 5, 0.25)
Y = np.arange(-5, 5, 0.25)
X, Y = np.meshgrid(X, Y)
Z = np.sqrt(X**2 + Y**2)

# Plot the surface.
surf = ax.plot_surface(X, Y, Z)

plt.show()