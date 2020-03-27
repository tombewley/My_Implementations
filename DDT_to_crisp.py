from my.DDT import DifferentiableDecisionTree

import numpy as np
import matplotlib.pyplot as plt
from joblib import load

dt = load('sin(16x).joblib')

Y, mus, ys = dt.predict([1], composition=True)

"""
Seems to be producing leaves whose values are way outside the distribution of the data, 
but 'cancel each other out' - this isn't a great way of solving the problem, and makes it hard to make crisp.
"""

print(Y)
print(mus)
print(ys)

# X = np.arange(0,2,0.02).reshape(-1,1)
# Y = dt.predict(X)
# plt.plot(X, Y)
# plt.show()