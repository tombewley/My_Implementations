from my.DDT import DifferentiableDecisionTree
import numpy as np
import matplotlib.pyplot as plt

BETA = 0.5
LR_Y = 0.5
LR_W = 0.5

c = 'br'
X = np.random.rand(100, 2)
y = (X[:,0] > X[:,1]).astype(int).reshape(-1,1)
X_test = np.array([[i,j] for i in np.linspace(0,1,10) for j in np.linspace(0,1,10)])

ddt = DifferentiableDecisionTree(depth=3, input_size=2, output_size=1, lr_y=LR_Y, lr_w=LR_W, y_lims=[[0],[1]], beta=BETA)

for i in range(10000):
    ddt.update_step(X, y, print_mae_before=True)

y_test = ddt.predict(X_test)
plt.imshow(y_test.reshape(10,10), origin='lower', extent=[0,1,0,1], cmap='RdBu', vmin=0, vmax=1)
plt.scatter(X[:,0], X[:,1], color=[c[i[0]] for i in y])

fig, ax = plt.subplots()
ddt.beta = 1000
y_test = ddt.predict(X_test)
plt.imshow(y_test.reshape(10,10), origin='lower', extent=[0,1,0,1], cmap='RdBu', vmin=0, vmax=1)
plt.scatter(X[:,0], X[:,1], color=[c[i[0]] for i in y])

plt.show()

