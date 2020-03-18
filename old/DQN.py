'''
Implementation of Deep Q-learning with experience replay (Mnih et al. Nature paper, 2015).
Forked and simplified from https://github.com/tokb23/dqn/blob/master/dqn.py.
'''

import os
import random
import numpy as np
import tensorflow as tf
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense # Convolution2D, Flatten


class Agent():
    def __init__(self,                                    # ---- ENVIRONMENTAL PROPERTIES ----
                 env_name,                                # Name
                 gamma,                                   # Temporal discount factor
                 num_features,                            # Size of observation space
                 num_actions,                             # Size of action space
                                                          # ---- MODEL PARAMETERS (DEFAULTS ARE FROM NATURE PAPER) ----
                 hidden_topology,                         # Number of units per hidden layer  
                 epsilon_schedule = (1.0, 1000000, 0.1),  # Degree of exploration in epsilon-greedy, which is linearly decremented during exploration (initial value, number of exploration steps, final value)
                 initial_replay_size = 20000,             # Number of random exploration steps to populate the replay memory with before training starts
                 full_replay_size = 1000000,              # Length of replay memory the agent uses for training
                 batch_size = 32,                         # Mini batch size used for SGD
                 train_interval = 4,                      # Number of timesteps between updates of Q network
                 target_update_interval = 10000,          # Number of timesteps between overwrites of target network 
                 save_interval = 10000,                   # Number of timesteps between checkpoint saves
                 learning_rate = 0.00025,                 # Learning rate used by RMSProp
                 momentum = 0.99,                         # Momentum used by RMSProp
                 min_grad = 1e-6,                         # Constant added to the squared gradient in the denominator of the RMSProp update
                 load_network = False                     # Whether to load latest checkpoint rather than starting training from scratch
                 ):
        
        self.env_name = env_name
        self.gamma = gamma
        self.num_features = num_features
        self.num_actions = num_actions
        self.hidden_topology = hidden_topology
        self.epsilon = epsilon_schedule[0]
        self.exploration_steps = epsilon_schedule[1]
        self.final_epsilon = epsilon_schedule[2]
        self.epsilon_step = (self.epsilon - self.final_epsilon) / self.exploration_steps
        self.initial_replay_size = initial_replay_size
        self.full_replay_size = full_replay_size
        self.batch_size = batch_size
        self.train_interval = train_interval
        self.target_update_interval = target_update_interval
        self.save_interval = save_interval
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.min_grad = min_grad

        # Parameters used for summary
        self.t = 0
        self.total_reward = 0
        self.total_q_max = 0
        self.total_loss = 0
        self.duration = 0
        self.episode = 0

        # Create replay memory
        self.replay_memory = deque()

        # Create q network
        self.s, self.q_values, q_network = self.build_network()
        self.q_network_weights = q_network.trainable_weights

        # Create target network
        self.st, self.target_q_values, target_network = self.build_network()
        target_network_weights = target_network.trainable_weights

        # Define target network update operation
        self.update_target_network = [target_network_weights[i].assign(self.q_network_weights[i]) for i in range(len(target_network_weights))]

        # Define loss and gradient update operation
        self.a, self.y, self.loss, self.grads_update = self.build_training_op()

        self.sess = tf.InteractiveSession()
        self.saver = tf.train.Saver(self.q_network_weights)
        self.summary_placeholders, self.update_ops, self.summary_op = self.setup_summary()
        self.summary_writer = tf.summary.FileWriter(self.env_name+'/summary', self.sess.graph)

        self.save_network_path = self.env_name+'/saved_networks'
        if not os.path.exists(self.save_network_path):
            os.makedirs(self.save_network_path)

        self.sess.run(tf.global_variables_initializer())

        # Load network
        if load_network:
            self.load_network()

        # Initialize target network
        self.sess.run(self.update_target_network)


    def build_network(self):
        model = Sequential()
        model.add(Dense(self.hidden_topology[0], activation='relu', input_shape=(self.num_features,)))#, kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.), ))
        for num_units in self.hidden_topology[1:]:
            model.add(Dense(num_units, activation='relu'))#, kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.)))
        model.add(Dense(self.num_actions))

        s = tf.placeholder(tf.float32, [None, self.num_features])
        q_values = model(s)

        return s, q_values, model


    def build_training_op(self):
        a = tf.placeholder(tf.int64, [None])
        y = tf.placeholder(tf.float32, [None])

        # Convert action to one hot vector
        a_one_hot = tf.one_hot(a, self.num_actions, 1.0, 0.0)
        q_value = tf.reduce_sum(tf.multiply(self.q_values, a_one_hot), reduction_indices=1)

        # Clip the error, the loss is quadratic when the error is in (-1, 1), and linear outside of that region
        error = tf.abs(y - q_value)
        quadratic_part = tf.clip_by_value(error, 0.0, 1.0)
        linear_part = error - quadratic_part
        loss = tf.reduce_mean(0.5 * tf.square(quadratic_part) + linear_part)

        optimizer = tf.train.RMSPropOptimizer(self.learning_rate, momentum=self.momentum, epsilon=self.min_grad)
        #optimizer = tf.train.AdamOptimizer()
        grads_update = optimizer.minimize(loss, var_list=self.q_network_weights)

        return a, y, loss, grads_update


    def get_action(self, observation):
        if self.epsilon >= random.random() or self.t < self.initial_replay_size:
            action = random.randrange(self.num_actions)
        else:
            action = np.argmax(self.q_values.eval(feed_dict={self.s: np.float32(observation).reshape((-1, self.num_features))}))

        # Anneal epsilon linearly over time
        if self.epsilon > self.final_epsilon and self.t >= self.initial_replay_size:
            self.epsilon -= self.epsilon_step

        return action


    def run(self, observation, action, reward, terminal, next_observation):

        # Clip all positive rewards at 1 and all negative rewards at -1, leaving 0 rewards unchanged
        reward = np.clip(reward, -1, 1)

        # Store transition in replay memory
        self.replay_memory.append((observation, action, reward, next_observation, terminal))
        if len(self.replay_memory) > self.full_replay_size:
            self.replay_memory.popleft()

        if self.t >= self.initial_replay_size:
            # Train network
            if self.t % self.train_interval == 0:
                self.train_network()

            # Update target network
            if self.t % self.target_update_interval == 0:
                self.sess.run(self.update_target_network)

            # Save network
            if self.t % self.save_interval == 0:
                save_path = self.saver.save(self.sess, self.save_network_path + '/' + self.env_name, global_step=self.t)
                print('Successfully saved: ' + save_path)

        self.total_reward += reward
        self.total_q_max += np.max(self.q_values.eval(feed_dict={self.s: np.float32(observation).reshape((-1, self.num_features))}))
        self.duration += 1

        if terminal:
            # Write summary
            if self.t >= self.initial_replay_size:
                stats = [self.total_reward, self.total_q_max / float(self.duration),
                        self.duration, self.total_loss / (float(self.duration) / float(self.train_interval))]
                for i in range(len(stats)):
                    self.sess.run(self.update_ops[i], feed_dict={
                        self.summary_placeholders[i]: float(stats[i])
                    })
                summary_str = self.sess.run(self.summary_op)
                self.summary_writer.add_summary(summary_str, self.episode + 1)

            # Debug
            if self.t < self.initial_replay_size:
                mode = 'random'
            elif self.initial_replay_size <= self.t < self.initial_replay_size + self.exploration_steps:
                mode = 'explore'
            else:
                mode = 'exploit'
            print('EPISODE: {0:6d} / TIMESTEP: {1:8d} / DURATION: {2:5d} / EPSILON: {3:.5f} / TOTAL_REWARD: {4:3.2f} / AVG_MAX_Q: {5:2.4f} / AVG_LOSS: {6:.5f} / MODE: {7}'.format(
                self.episode + 1, self.t, self.duration, self.epsilon,
                self.total_reward, self.total_q_max / float(self.duration),
                self.total_loss / (float(self.duration) / float(self.train_interval)), mode))

            self.total_reward = 0
            self.total_q_max = 0
            self.total_loss = 0
            self.duration = 0
            self.episode += 1

        self.t += 1


    def train_network(self):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        terminal_batch = []
        y_batch = []

        # Sample random minibatch of transition from replay memory
        minibatch = random.sample(self.replay_memory, self.batch_size)
        for data in minibatch:
            state_batch.append(data[0])
            action_batch.append(data[1])
            reward_batch.append(data[2])
            next_state_batch.append(data[3])
            terminal_batch.append(data[4])

        # Convert True to 1, False to 0
        terminal_batch = np.array(terminal_batch) + 0

        target_q_values_batch = self.target_q_values.eval(feed_dict={self.st: np.float32(np.array(next_state_batch))})
        y_batch = reward_batch + (1 - terminal_batch) * self.gamma * np.max(target_q_values_batch, axis=1)

        loss, _ = self.sess.run([self.loss, self.grads_update], feed_dict={
            self.s: np.float32(np.array(state_batch)),
            self.a: action_batch,
            self.y: y_batch
        })

        self.total_loss += loss


    def setup_summary(self):
        episode_total_reward = tf.Variable(0.)
        tf.summary.scalar(self.env_name + '/Total Reward/Episode', episode_total_reward)
        episode_avg_max_q = tf.Variable(0.)
        tf.summary.scalar(self.env_name + '/Average Max Q/Episode', episode_avg_max_q)
        episode_duration = tf.Variable(0.)
        tf.summary.scalar(self.env_name + '/Duration/Episode', episode_duration)
        episode_avg_loss = tf.Variable(0.)
        tf.summary.scalar(self.env_name + '/Average Loss/Episode', episode_avg_loss)
        summary_vars = [episode_total_reward, episode_avg_max_q, episode_duration, episode_avg_loss]
        summary_placeholders = [tf.placeholder(tf.float32) for _ in range(len(summary_vars))]
        update_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in range(len(summary_vars))]
        summary_op = tf.summary.merge_all()

        return summary_placeholders, update_ops, summary_op


    def load_network(self):
        checkpoint = tf.train.get_checkpoint_state(self.save_network_path)
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print('Successfully loaded: ' + checkpoint.model_checkpoint_path)
        else:
            print('Training new network...')


    def get_action_at_test(self, observation):
        q_values = self.q_values.eval(feed_dict={self.s: np.float32(observation).reshape((-1, self.num_features))})
        action = np.argmax(q_values)
        self.t += 1
        return action, q_values