from my.DT import DecisionTreeClassifier
import numpy as np

X = np.random.rand(100000, 2)
y = X[:,0] > X[:,1]

dt = DecisionTreeClassifier(max_depth=10)


dt.fit(X, y, feature_names=['a','b'])#, sample_weight=np.random.rand(100000))

# print(dt.pwl)


dt.to_code()

