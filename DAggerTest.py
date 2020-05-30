from DAgger import DAgger   

class SimpleEnv:
    def __init__(self): return
    def reset(self): self.t = 0; return 1
    def step(self, action): 
        self.t += 1
        if self.t >= 20: done = True
        else: done = False
        return [1,2,3], 1, done, 1 
class SimpleAgent:
    def __init__(self, action): self.action = action
    def act(self, observation): return self.action

target = SimpleAgent(0)
clone = SimpleAgent(1)

DAgger = DAgger(SimpleEnv(), target, verbose=True)

DAgger.iterate(clone, num_episodes=3, beta=0.5)

X, y = DAgger.D()

print(y)
