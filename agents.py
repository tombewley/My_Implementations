from joblib import load

# =============================================================================
# Decision tree; sklearn or my.

class DTAgent:
    def __init__(self, mode='my', path_to_tree=None, multiagent=False, ix_offset=0, **kwargs):
        self.ix_offset = ix_offset
        self.mode = mode
        self.multiagent = multiagent
        if path_to_tree==None:
            if self.mode == 'my': from my.DT import DecisionTreeClassifier
            elif self.mode == 'sk': from sklearn.tree import DecisionTreeClassifier
            self.dt = DecisionTreeClassifier(**kwargs)
        else:
            self.dt = load(path_to_tree)

    def act(self, observation):
        if self.multiagent:
            actions = {}
            for a, X in observation.items():
                if self.mode == 'my':
                    actions[a] = self.dt.predict(X) + self.ix_offset
                elif self.mode == 'sk':
                    actions[a] = self.dt.predict(X.reshape(1,-1))[0] + self.ix_offset
            return actions
        else:
            if self.mode == 'my':
                return self.dt.predict(observation) + self.ix_offset
            elif self.mode == 'sk':
                return self.dt.predict(observation.reshape(1,-1))[0] + self.ix_offset 