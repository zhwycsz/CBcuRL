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


from continuous_agent import *


def learn():

    debug = True

    matplotlib.rcParams.update({'font.size': 22})
    if debug: print('NEURAL')

    # open parameter file
    f = open('/Users/Neythen/Desktop/masters_project/app/CBcurl_master/examples/parameter_files/double_auxotroph.yaml')
    param_dict = yaml.load(f)
    f.close()

    validate_param_dict(param_dict)
    param_dict = convert_to_numpy(param_dict)

    # extract parameters
    NUM_EPISODES, test_freq, explore_denom, step_denom, T_MAX,MIN_STEP_SIZE, \
        MAX_STEP_SIZE, MIN_EXPLORE_RATE, MAX_EXPLORE_RATE, cutoff, hidden_layers, buffer_size  = param_dict['train_params']
    NOISE, error = param_dict['noise_params']
    num_species, num_controlled_species, num_N_states, N_bounds, num_Cin_states, Cin_bounds = \
        param_dict['Q_params'][0], param_dict['Q_params'][1],  param_dict['Q_params'][2], param_dict['Q_params'][7], param_dict['Q_params'][4], param_dict['Q_params'][5]
    ode_params = param_dict['ode_params']
    Q_params = param_dict['Q_params'][0:7]
    initial_X = param_dict['Q_params'][7]
    initial_C = param_dict['Q_params'][8]
    initial_C0 = param_dict['Q_params'][9]



    tf.reset_default_graph() #clear the tensorflow graph.

    #initialise saver and tensorflow graph

    init = tf.global_variables_initializer()


    # make directories to store results
    """
    os.makedirs(os.path.join(save_path,'data','train'), exist_ok = True)
    os.makedirs(os.path.join(save_path,'graphs','train'), exist_ok = True)
    os.makedirs(os.path.join(save_path,'saved_network'), exist_ok = True)
    os.makedirs(os.path.join(save_path,'state_actions'), exist_ok = True)
    """

    with tf.Session() as sess:
        K.set_session(sess)

        batch_size = 10
        tau = 0.1
        learning_rate = 1
        actor = ActorNetwork(sess, batch_size, tau, learning_rate)
        critic = CriticNetwork(sess, batch_size, tau, learning_rate)
        buffer = ExperienceBuffer(100)
        agent = Agent(sess, actor, critic, buffer, Cin_bounds[1])

        episode_ts, episode_rewards, time_avs, rewards_avs = [], [], [], []

        # fill buffef with experiences based on random actions
        i = 0
        while len(agent.buffer.buffer) < 100:
            #reset
            X = np.random.random([num_species, ])* N_bounds[1]
            C = initial_C
            C0 = initial_C0


            i += 1
            for t in range(T_MAX):
                X, C, C0 = agent.pre_train_step(sess, X, C, C0, t, Q_params, ode_params)

                if (not all(x>cutoff for x in X)) or t == T_MAX - 1: # if done
                    break

        print('BUFFER DONE')
        nIters = 0 # keep track for updating target network

        for episode in range(1,NUM_EPISODES+1):

            # reset for this episode
            X = np.random.random([num_species, ]) * N_bounds[1]

            C = initial_C
            C0 = initial_C0
            xSol = np.array([X])
            running_reward = 0
            ep_history = np.array([[]])
            explore_rate = get_rate(episode, MIN_EXPLORE_RATE, MAX_EXPLORE_RATE, explore_denom)
            step_size = get_rate(episode, MIN_STEP_SIZE, MAX_STEP_SIZE, step_denom)

            # run episode
            for t in range(T_MAX):
                nIters += 1 # for target Q update
                X, C, C0, xSol_next, reward = agent.train_step(X, C, C0, explore_rate, Q_params, ode_params, t)

                if NOISE:
                    X = add_noise(X, error)

                running_reward += reward

                xSol = np.append(xSol, xSol_next, axis = 0)

                if (not all(x>cutoff for x in X)) or t == T_MAX - 1: # if done
                    break


            # track results
            if episode%test_freq == 0 and episode != 0:

                if debug:
                    print('Episode: ', episode)
                    print('Explore rate: ' + str(explore_rate))
                    print('Step size', step_size)
                    print('Average Time steps: ', np.mean(episode_ts))
                    print('Average Reward: ', np.mean(episode_rewards))
                    print()

                # add current results
                episode_rewards.append(running_reward)
                episode_ts.append(t)

                time_avs.append(np.mean(episode_ts))
                print(episode_ts)
                rewards_avs.append(np.mean(episode_rewards))

                # reset
                train_reward = running_reward
                episode_ts = []
                episode_rewards = []

                """
                if debug:
                    # plot current population curves
                    plt.figure(figsize = (22.0,12.0))
                    plot_pops(xSol, os.path.join(save_path,'WORKING_graphs','train','pops_train_' + str(int(episode/test_freq)) + '.png'))
                    np.save(os.path.join(save_path,'WORKING_data','train','pops_train_' + str(int(episode/test_freq)) + '.npy'), xSol)
                """
            else:
                episode_rewards.append(running_reward)
                episode_ts.append(t)


if __name__ == '__main__':
    learn()
