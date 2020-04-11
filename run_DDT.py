from my.DDT import DifferentiableDecisionTree

import numpy as np
import pandas as pd
from joblib import dump
import matplotlib.pyplot as plt

FNAME = 'hf_hf_nobalance_band'

DEPTH = 3
W_INIT_SD = 0.1
NUM_UPDATES = 10000
BATCH_SIZE = 64
LR_Y = 0.05 # Larger tree = higher learning rate.
LR_W = 0.01

# --------------------------------------------

df = pd.read_csv('{}.csv'.format(FNAME))
X = np.array(df[['speed','me_to_junc','me_to_inf','x1_vs_me','inf_rel_speed','x1_rel_speed']])
T = np.array(df['acc_label']).reshape(-1,1)

print(np.shape(X), np.shape(T))

# # 2 dimensional.
# # X = np.array([np.random.uniform([-5,-5],[5,5]) for _ in range(num_instances)])
# # T = np.sqrt(X[:,0]**2 + X[:,1]**2).reshape(-1,1)

ddt = DifferentiableDecisionTree(depth = DEPTH, 
                                input_size = np.shape(X)[1], 
                                output_size = np.shape(T)[1], 
                                lr_y = LR_Y, 
                                lr_w = LR_W, 
                                y_lims = (np.min(T, axis=0), np.max(T, axis=0)),
                                )

fig, ax = plt.subplots()
plt.ion(); plt.show()

mae_ma_last = np.inf
mae = [np.inf] * 100
for i in range(NUM_UPDATES):
    # Perform gradient update on a batch.
    idx = np.random.randint(np.shape(X)[0], size=BATCH_SIZE) 
    X_batch = X[idx,:]; T_batch = T[idx,:]
    ddt.update_step(X_batch, T_batch)
    # Compute mean absolute error after update.
    mae[i % 100] = np.abs(T_batch - ddt.predict(X_batch)).mean()

    if i % 10 == 0: 
        mae_ma = np.mean(mae)
        ax.plot([i-10, i], [mae_ma_last, mae_ma], 'k')
        plt.pause(0.001)
        mae_ma_last = mae_ma
    print(i, mae_ma)
    print([round(leaf.y[0], 2) for leaf in ddt.leaves])

    #if mae < 0.2: break

dump(ddt, '{}_D={}_{}.joblib'.format(FNAME, DEPTH, round(mae_ma,2)))