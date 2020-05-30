"""
Generic pipeline for Dataset Aggregation (DAgger).
Originally introducted in:  Ross, Stéphane, Geoffrey Gordon, and Drew Bagnell. "A reduction of imitation learning and structured prediction to no-regret online learning". 
                            Proceedings of the Fourteenth International Conference on Artificial Intelligence and Statistics. 2011.

Also able to handle AggreVaTe, an extension that makes use of cost-to-go information.
This is from:               Ross, Stephane, and J. Andrew Bagnell. "Reinforcement and Imitation Learning via Interactive No-Regret Learning". 
                            ArXiv:1406.5979 [Cs, Stat], 23 June 2014. http://arxiv.org/abs/1406.5979.


TODO: Implement AggreVaTe. 
    - Note that it's only clear how to do so with fixed-length episides with no early termination.
    - For infinite episodes, there's no bounded range for the interruption time t.
        - Big problem.
    - For early-terminating episodes, we might stop before we reach the interruption time.
        - Smaller problem; could just discard episodes or store all rewards and pick t after the fact.

Assumes we have:

    An environment with basic OpenAI Gym-like functionality:
        observation                     = env.reset() 
        observation, reward, done, info = env.step(action)

    A stationary target policy, passed in at initialisation, with an "act" method:
        action = target.act(observation)

    A nonstationary imitator policy also with an "act" method:
        action = imitator.act(observation)

The actual online learning process to generate each imitator policy happens externally.

"""

import numpy as np
from tqdm import tqdm

class DAgger:

    def __init__(self, env, target, verbose=False):
        self.env = env
        self.target = target
        self.verbose = verbose
        self.D = Dataset() # DAgger dataset.
        self.DA = Dataset(AggreVaTe=True) # AggreVaTe dataset.
        self.n = 0 # DAgger iteration counter.
        self.nA = 0 # AggreVaTe iteration counter.

# ===================================================================================================================

    def iterate(self, imitator, AggreVaTe=False, num_samples=0, num_episodes=0, complete_episodes=False, beta=None):
        """Complete a round of the iterative data collection process for DAgger or AggreVaTe."""

        if AggreVaTe: 
            # AggreVaTe mode.
            self.nA += 1

        else:
            # DAgger mode.
            self.n += 1 
            assert (num_samples > 0 and num_episodes == 0) or (num_samples == 0 and num_episodes > 0) # Must specify either num episodes or num samples, and not both.
            if beta == None: beta = int(self.n==1) # If no beta value specified, follow the recommended approach: 1 if n = 1, 0 otherwise.
            else: assert beta >= 0 and beta <= 1 # Beta needs to be in [0,1].
            if self.verbose: print('DAgger iteration {}: collecting {}{} with beta = {}.'.format(self.n, 
                                                                                                'a minimum of ' if num_episodes == 0 and complete_episodes == True else '',
                                                                                                '{} episodes'.format(num_episodes) if num_samples == 0 else '{} samples'.format(num_samples), 
                                                                                                beta))     

        iteration_complete = False; ep = 0; m = 0
        with tqdm(total=(num_samples if num_episodes==0 else num_episodes)) as pbar:
            while not iteration_complete:
                ep += 1
                #if self.verbose: print('   Episode {}'.format(ep))

                # Initialise a new episode.
                observation = self.env.reset(); done = False
                while not done:
                    # Get target action.
                    target_action = self.target.act(observation)
                    # If a random number exceeds beta, use imitator action. 
                    if np.random.rand() > beta: action = imitator.act(observation)
                    # Otherwise, use target action.
                    else: action = target_action
                    # Store sample with target action.
                    self.D._X.append(observation)
                    self.D._y.append(target_action)

                    m += 1
                    # Step environment.
                    observation, reward, done, _ = self.env.step(action)
              
                    # Check stopping criteria.
                    if num_samples > 0:
                        pbar.update()
                        if m >= num_samples: 
                            iteration_complete = True
                            if not complete_episodes: break
                if num_episodes > 0:
                    pbar.update()
                    if ep >= num_episodes:
                        iteration_complete = True

        if self.verbose: print('Dataset now has {} samples in total.'.format(len(self.D._X)))

    
    def reset(self):
        self.D = Dataset(); self.n = 0

    
    def evaluate(self, num_episodes, gamma, imitator=None):
        """Roll out a policy in the environment, without collecting data for learning.
        Return per-episode returns, with discount factor applied.
        Uses the target policy by default unless an imitator is provided."""

        ep_returns = [0] * num_episodes
        #for ep in range(num_episodes):
        for ep in tqdm(range(num_episodes)):
            observation = self.env.reset(); done = False; t = 0
            while not done:
                if imitator == None: action = self.target.act(observation)
                else: action = imitator.act(observation)
                observation, reward, done, _ = self.env.step(action)
                ep_returns[ep] += reward * (gamma**t)
                t += 1
        return ep_returns

# ===================================================================================================================

class Dataset:
    
    def __init__(self, AggreVaTe=False):
        self._X = []
        self._y = []
        self.AggreVaTe = AggreVaTe
        if self.AggreVaTe: self._Q = []

    def __call__(self):
        if self.AggreVaTe: return np.array(self._X), np.array(self._y), np.array(self._Q)
        else:              return np.array(self._X), np.array(self._y)

    def __str__(self):
        if self.AggreVaTe: return 'AggreVaTe dataset with {} samples.'.format(len(self._X))
        else:              return 'DAgger dataset with {} samples.'.format(len(self._X))