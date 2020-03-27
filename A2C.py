'''
Code for (Asynchronous) Advantage Actor-Critic.
Adapted from: http://inoryy.com/post/tensorflow2-deep-reinforcement-learning/
'''


import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as kl
import tensorflow.keras.losses as kls
import tensorflow.keras.optimizers as ko


class ProbabilityDistribution(tf.keras.Model):
    def call(self, logits, **kwargs):
        # Sample a random categorical action from the given logits.
        return tf.squeeze(tf.random.categorical(logits, 1), axis=-1)

# ------------------------------------------------------------

class Model(tf.keras.Model):
    def __init__(self, input_shape, num_actions):
        # Inherits from keras.Model. Initialise with the argument 'mlp_policy'.
        super().__init__('mlp_policy')
        # Define network structure.
        self.hidden1 = kl.Dense(128, activation='relu')
        self.hidden2 = kl.Dense(128, activation='relu')
        self.value = kl.Dense(1, name='value')
        # Logits are unnormalized log probabilities.
        self.logits = kl.Dense(num_actions, name='policy_logits')
        self.dist = ProbabilityDistribution()
        # Build.
        self.in_shape = input_shape
        self.build(input_shape=input_shape)


    def call(self, inputs, **kwargs):
        # Inputs is a numpy array, convert to a tensor.
        x = tf.convert_to_tensor(inputs)
        # Separate hidden layers from the same input tensor.
        hidden_logs = self.hidden1(x)
        hidden_vals = self.hidden2(x)
        return self.logits(hidden_logs), self.value(hidden_vals)


    # def __init__(self, input_shape, num_actions):
    #     # Inherits from keras.Model. Initialise with the argument 'mlp_policy'.
    #     super().__init__('mlp_policy')

    #     # Define network structures.
    #     self.actor = tf.keras.models.Sequential()
    #     self.actor.add(kl.Dense(128, activation='relu'))
    #     self.actor.add(kl.Dense(num_actions, name='policy_logits'))
    #     self.critic = tf.keras.models.Sequential()
    #     self.critic.add(kl.Dense(128, activation='relu'))
    #     self.critic.add(kl.Dense(1, name='value'))

    #     self.dist = ProbabilityDistribution()

    #     self.in_shape = input_shape
    #     self.build(input_shape=input_shape)


    # def call(self, inputs, **kwargs):
    #     # Inputs is a numpy array, convert to a tensor.
    #     x = tf.convert_to_tensor(inputs)
    #     return self.actor(x), self.critic(x)


    def action_value(self, obs):
        # Executes `call()` under the hood.
        logits, value = self.predict_on_batch(obs)
        action = self.dist.predict_on_batch(logits)
        return np.squeeze(action, axis=-1), np.squeeze(value, axis=-1)

# ------------------------------------------------------------

class Agent:
    def __init__(self, model, learning_rate=7e-3, gamma=0.99, batch_size=64, value_c=0.5, entropy_c=1e-4):
        self.model = model
        self.gamma = gamma
        self.batch_size = batch_size
        self.value_c = value_c # Coefficients are used for the loss terms.
        self.entropy_c = entropy_c

        # Define optimiser and losses used to train the model.
        self.model.compile(    optimizer=ko.RMSprop(lr=learning_rate),
                               #optimizer=ko.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, amsgrad=False),
                               loss=[self._logits_loss, self._value_loss])

        # Structures for storing data used during training.
        self.actions = np.empty((batch_size,), dtype=np.int32)
        self.rewards, self.dones, self.values = np.empty((3, batch_size))
        self.observations = np.empty((batch_size,) + self.model.in_shape)   
        self.ep_rewards = [0.0]
        self.t = 0 # Total timesteps seen.
        self.i = 0 # Counter used to determine when to train (batch size reached).  
        self.update_count = 0 # Number of updates completed. 


    def run(self, obs, action, value, next_obs, reward, done):
        
        # Store values.
        self.observations[self.i] = obs
        self.actions[self.i] = action
        self.values[self.i] = value
        self.rewards[self.i] = reward
        self.ep_rewards[-1] += reward
        self.dones[self.i] = done
        if done: self.ep_rewards.append(0.0)
        
        self.t += 1; self.i += 1

        # If have reached the end of a batch, perform a full training step on the collected batch.
        if self.i == self.batch_size: 
            self.i = 0; self.update_count += 1

            _, next_value = self.model.action_value(next_obs[None, :])

            returns, advs = self._returns_advantages(self.rewards, self.dones, self.values, next_value)
            acts_and_advs = np.concatenate([self.actions[:, None], advs[:, None]], axis=-1)

            # Return losses.
            return self.model.train_on_batch(self.observations, [acts_and_advs, returns])


    def _returns_advantages(self, rewards, dones, values, next_value):
        # `next_value` is the bootstrap value estimate of the future state (critic).
        returns = np.append(np.zeros_like(rewards), next_value, axis=-1)

        # Returns are calculated as discounted sum of future rewards.
        for t in reversed(range(rewards.shape[0])):
            returns[t] = rewards[t] + self.gamma * returns[t + 1] * (1 - dones[t])
        returns = returns[:-1]

        # Advantages are equal to returns - baseline (value estimates in our case).
        advantages = returns - values

        return returns, advantages


    def _value_loss(self, returns, value):
        # Value loss is typically MSE between value estimates and returns.
        return self.value_c * kls.mean_squared_error(returns, value)

    
    def _logits_loss(self, actions_and_advantages, logits):
        # A trick to input actions and advantages through the same API.
        actions, advantages = tf.split(actions_and_advantages, 2, axis=-1)

        # Sparse categorical CE loss obj that supports sample_weight arg on `call()`.
        # `from_logits` argument ensures transformation into normalized probabilities.
        weighted_sparse_ce = kls.SparseCategoricalCrossentropy(from_logits=True)

        # Policy loss is defined by policy gradients, weighted by advantages.
        # Note: we only calculate the loss on the actions we've actually taken.
        actions = tf.cast(actions, tf.int32)
        policy_loss = weighted_sparse_ce(actions, logits, sample_weight=advantages)

        # Entropy loss can be calculated as cross-entropy over itself.
        probs = tf.nn.softmax(logits)
        entropy_loss = kls.categorical_crossentropy(probs, probs)

        # We want to minimize policy and maximize entropy losses.
        # Here signs are flipped because the optimizer minimizes.
        return policy_loss - self.entropy_c * entropy_loss