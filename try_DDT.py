from joblib import load
import numpy as np


ddt = load('sc_hf_nobalance_eight_D=5_0.58.joblib')

Y, mus, ys = ddt.predict(
    #[0.1,3.3773727662002644,0.987107069650536,3.1027183174620285,0.0,0.0], # 4
    [0.02996170423925227,0.6620927391935444,1.3398205270118422,0.4833355998995684,-0.06533440752528236,-0.048788272690402514], # 1
    #[0.024306684019500174,0.8993194862013251,0.7189478192926585,0.41126866386690186,-0.050427617274109354,-0.006692368484406078], # 3
    composition=True)

np.set_printoptions(precision=3, suppress=True)
print('\n')
print('y =   {}'.format(Y[0]))
print('mus = {}'.format(mus[0]))
print('ys =  {}'.format(np.transpose(ys)[0]))
print('\n')
