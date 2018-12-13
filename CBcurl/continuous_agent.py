from collections import deque
import random
import numpy as np
import keras
from keras import backend as K
import matplotlib
from utilities import *
import yaml
import tensorflow as tf
import os
import sys
from keras.layers import Dense, Flatten, Input, merge, Lambda
from keras.models import Sequential, Model
from keras.optimizers import Adam

class ExperienceBuffer():
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = deque()

    def add(self, experience):
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(experience)
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def size(self):
        return len(self.buffer)

    def sample_batch(self, batch_size):
        '''
        batch_size specifies the number of experiences to add
        to the batch. If the replay buffer has less than batch_size
        elements, simply return all of the elements within the buffer.
        Generally, you'll want to wait until the buffer has at least
        batch_size elements before beginning to sample from it.
        '''
        batch = []

        if len(self.buffer) < batch_size:
            batch = random.sample(self.buffer, len(self.buffer))
        else:
            batch = random.sample(self.buffer, batch_size)

        s_batch = np.array([experience[0] for experience in batch])
        a_batch = np.array([experience[1] for experience in batch])
        r_batch = np.array([experience[2] for experience in batch])
        s1_batch = np.array([experience[3] for experience in batch])

        return s_batch, a_batch, r_batch, s1_batch

    def clear(self):
        self.buffer.clear()


class Network():
    def update_target(self):
        # update target network to the primary networks weights
        primary_vars = self.model.get_weights()
        target_vars = self.target.get_weights()

        for i in range(len(primary_vars)):
            target_vars[i] = self.tau * primary_vars[i] + (1 - self.tau)* target_vars[i]
        self.target.set_weights(target_vars)


class ActorNetwork(Network):

    def __init__(self, sess, batch_size, tau, learning_rate):
        self.state_size = 2
        self.action_size = 2
        self.sess = sess
        self.model = self.create_network()
        self.target_model = self.create_network()
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.tau = tau
        # make networks
        self.model = self.create_network()
        self.target = self.create_network()

        self.action_gradient = tf.placeholder(tf.float32,[None, self.action_size])
        self.params_grad = tf.gradients(self.model.output, self.model.trainable_weights, -self.action_gradient)
        grads = zip(self.params_grad, self.model.trainable_weights)
        self.optimize = tf.train.AdamOptimizer(learning_rate).apply_gradients(grads)

        self.sess.run(tf.initialize_all_variables())



    def train(self, states, action_grads):
        self.sess.run(self.optimize, feed_dict={
            self.model.input: states,
            self.action_gradient: action_grads
        })

    def create_network(self):

        model = tf.keras.models.Sequential([
          tf.keras.layers.Dense(20, input_shape = ([self.state_size]), activation = tf.nn.relu, name = 'actor_in'),
          tf.keras.layers.Dense(20, activation=tf.nn.relu),
          tf.keras.layers.Dense(20, activation=tf.nn.relu),
          tf.keras.layers.Dense(20, activation=tf.nn.relu),
          tf.keras.layers.Dropout(0.2),
          tf.keras.layers.Dense(self.action_size, name = 'actor_out', activation = 'sigmoid') # linear for now, try sigmoid with a scaling factor
        ])

        return model
        # use model.fit(), model.predict()

    def train(self, states, action_grads):


        # This gradient will be provided by the critic network
        self.action_gradient = tf.placeholder(tf.float32, [None, self.action_size])

        # Combine the gradients, dividing by the batch size to
        # account for the fact that the gradients are summed over the
        # batch by tf.gradients
        self.unnormalized_actor_gradients = tf.gradients(
            self.model.output, self.model.trainable_weights, -self.action_gradient)
        self.actor_gradients = list(map(lambda x: tf.div(x, self.batch_size), self.unnormalized_actor_gradients))

        # Optimization Op
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).\
            apply_gradients(zip(self.actor_gradients, self.model.trainable_weights))


class CriticNetwork(Network):

    def __init__(self, sess, batch_size, tau, learning_rate):
        self.state_size = 2
        self.action_size = 2
        self.sess = sess

        self.batch_size = batch_size
        self.tau = tau
        self.learning_rate = learning_rate
        self.model, self.state, self.action = self.create_network()
        self.target, self.target_state, self.target_action = self.create_network()
        # Network target (y_i)
        # Obtained from the target networks
        self.predicted_q_value = tf.placeholder(tf.float32, [None, 1])

        # Get the gradient of the net w.r.t. the action
        self.action_grads = tf.gradients(self.model.output, self.action)

        self.sess.run(tf.initialize_all_variables())

    def create_network(self):

        state = Input(shape=[self.state_size], name = 'critic_state_in')
        action = Input(shape=[self.action_size],name='critic_action_in')
        w1 = Dense(20, activation='relu')(state)
        a1 = Dense(20, activation='linear')(action)
        h1 = Dense(20, activation='linear')(w1)
        h2 = merge([h1,a1],mode='sum')
        h3 = Dense(20, activation='relu')(h2)
        V = Dense(self.action_size,activation='linear')(h3)
        model = Model(input=[state,action],output=V)
        adam = Adam(lr=self.learning_rate)
        model.compile(loss='mse', optimizer=adam)

        return model, state, action



    def gradients(self, states, actions):
        return self.sess.run(self.action_grads, feed_dict={self.state: states, self.action: actions})[0]

class Agent():

    def __init__(self, sess, actor, critic, buffer, action_scaling):
        self.actor = actor
        self.critic = critic
        self.buffer = buffer
        self.sess = sess
        self.action_scaling = action_scaling

    def get_next_solution(self, Cin, X, C, C0, Q_params, ode_params,t):

        num_species, _, _, _, _, _, _ = Q_params

        # create state vector
        S = np.append(X, C)
        S = np.append(S, C0)



        time_diff = 4 # frame skipping
        sol = odeint(sdot, S, [t + x*1 for x in range(time_diff)], args=(Cin, ode_params, num_species))[1:]


        # extract information from solution
        xSol = sol[:, 0:num_species]
        X1 = sol[-1, 0:num_species]
        C1 = sol[-1, num_species:-1]
        C01 = sol[-1, -1]

        assert len(Cin) == num_species, 'Cin is the wrong length: ' + str(len(Cin))
        assert len(X1) == num_species, 'X is the wrong length: ' + str(len(X))
        assert len(C1) == num_species, 'C is the wrong length: ' + str(len(C1))

        return X1, C1, C01, xSol

    def pre_train_step(self, sess, X, C, C0, t, Q_params, ode_params):
        '''
        Carries out one random training step to gather experience for the experience buffer when using DQN
        Parameters:
            X: array storing the populations of each bacteria
            C: array contatining the concentrations of each rate limiting nutrient
            C0: the concentration of the common carbon source at this time point
            t: the current time
            Q_params: learning parameters
            ode_params: parameters for the numerical solver odeint
        Returns:
            X1: populations at next timestep
            C1: concentrations of auxotrophc nutrient at each time step
            C01: concentration of the carbon source at next time point
        '''

        num_species, num_controlled_species, num_N_states, N_bounds, num_Cin_states, Cin_bounds, gamma = Q_params # extract parameters

        Cin = np.random.random((2,)) * Cin_bounds[1]


        S = np.append(X, C)
        S = np.append(S, C0)

        # get next time step
        time_diff = 4  # frame skipping
        sol = odeint(sdot, S, [t + x *1 for x in range(time_diff)], args=(Cin, ode_params, num_species))[1:]


        # extract information from sol
        xSol = sol[:, 0:num_species]
        X1 = sol[-1, :num_species]
        C1 = sol[-1, num_species:-1]
        C01 = sol[-1, -1]


        assert len(Cin) == num_species, 'Cin is the wrong length: ' + str(len(Cin))
        assert len(X1) == num_species, 'X is the wrong length: ' + str(len(X))
        assert len(C1) == num_species, 'C is the wrong length: ' + str(len(C1))

        reward = self.reward(X1)

        self.buffer.add([X, Cin, reward, X1])

        return X1, C1, C01

    def train_step(self, X, C, C0, explore_rate, Q_params, ode_params, t):
        # select action from actor plus some exploration noise

        Cin = self.actor.model.predict(X.reshape(-1,2))[0]
        Cin *= self.action_scaling

        if np.random.random() < explore_rate: # maybe switch to noise
            Cin = np.random.random((2,)) * 0.1
        # run next step of the simulation
        X1, C1, C01, xSol_next = self.get_next_solution(Cin, X, C, C0, Q_params, ode_params, t)

        reward = self.reward(X1)

        self.buffer.add([X, Cin, reward, X1])

        # train on batch of experiences
        s_batch, a_batch, r_batch, s1_batch = self.buffer.sample_batch(2)

        #build TD target
        target_as = self.actor.target.predict(s1_batch)
        target_Qs = self.critic.target.predict([s1_batch, target_as]) #CHECK APPENDS ARE WORKING PROPERLY ON THE BATCH
        y = 0.9
        TD_target = reward + y * target_Qs

        # update critic network
        self.critic.model.train_on_batch([s_batch, a_batch], TD_target)
        action_grads = self.critic.gradients(s_batch, self.actor.model.predict(s_batch))
        self.actor.train(s_batch, action_grads)

        #update target networks
        self.actor.update_target()
        self.critic.update_target()
        
        return X1, C1, C01, xSol_next, reward

    def reward(self, X):
        if all(x > 2 for x in X):
            reward = 1
        else:
            reward = -1
        '''
        if 100 < X[0] < 400 and 400 < X[1] < 700:
            reward = 1
        else:
            reward = -1
        if any(x < 1/1000 for x in X):
            reward = - 10
        '''
        return reward
