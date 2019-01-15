from collections import deque
import random
import numpy as np
import keras
from keras import backend as K
import matplotlib
matplotlib.use('agg')
from utilities import *
import yaml
import tensorflow as tf
import os
import sys
from keras.layers import Dense, Flatten, Input, merge, Lambda
from keras.models import Sequential, Model
from keras.optimizers import Adam
import time
import gc

from continuous_agent import *
import matplotlib.pyplot as plt

from plot_funcs import *

def learn(r):

    debug = True

    matplotlib.rcParams.update({'font.size': 22})
    if debug: print('NEURAL')

    # open parameter file
    f = open('/Users/Neythen/Desktop/masters_project/app/CBcurl_master/examples/parameter_files/double_auxotroph.yaml')
    #f = open('/jmain01/home/JAD012/cpb03/njt97-cpb03/CBcurl_master/examples/parameter_files/double_auxotroph.yaml')
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
    save_path = '/Users/Neythen/Desktop/masters_project/results/continuous'
    #save_path = '/jmain01/home/JAD012/cpb03/njt97-cpb03/Job_Scripts/continuous/repeat' + str(r)
    # make directories to store results

    os.makedirs(os.path.join(save_path,'data','train'), exist_ok = True)
    os.makedirs(os.path.join(save_path,'graphs','train'), exist_ok = True)
    os.makedirs(os.path.join(save_path,'saved_network'), exist_ok = True)
    os.makedirs(os.path.join(save_path,'state_actions'), exist_ok = True)

    with tf.Session() as sess:
        K.set_session(sess)

        batch_size = 10
        tau = 0.001
        learning_rate = 0.0001
        critic = CriticNetwork(sess, batch_size, tau, learning_rate)
        learning_rate = 0.00001
        actor = ActorNetwork(sess, batch_size, tau, learning_rate, Cin_bounds[1])

        buffer = ExperienceBuffer(buffer_size)
        agent = Agent(sess, actor, critic, buffer)
        episode_ts, episode_rewards, time_avs, rewards_avs = [], [], [], []

        # fill buffer with experiences based on random actions
        i = 0
        while len(agent.buffer.buffer) < buffer_size:
            #reset
            X = np.random.random([num_species, ])* N_bounds[1]
            C = initial_C
            C0 = initial_C0

            i += 1
            for t in range(T_MAX):
                X, C, C0 = agent.pre_train_step(sess, X, C, C0, t, Q_params, ode_params, cutoff)

                if (not all(x>cutoff for x in X)) or t == T_MAX - 1: # done check here and inside agent, remove one
                    break

        print('BUFFER DONE')
        nIters = 0 # keep track for updating target network

        for episode in range(1,NUM_EPISODES+1):

            # reset for this episode
            #X = np.random.random([num_species, ]) * N_bounds[1]
            X = initial_X
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

                X, C, C0, xSol_next, reward, Cin_nw, C_in, action_grads = agent.train_step(X, C, C0, explore_rate, Q_params, ode_params, t, episode%test_freq == 0, cutoff)

                if NOISE:
                    X = add_noise(X, error)

                running_reward += reward

                xSol = np.append(xSol, xSol_next, axis = 0)

                if (not all(x>cutoff for x in X)) or t == T_MAX - 1: # if done
                    break
            print(X)
            print('Cin_nw: ',Cin_nw)
            print('C_in: ', C_in)
            #print(action_grads)
            #print(X, C, C0)

            #print('actor_weights: ', agent.actor.model.get_weights())
            print()


            #print('critic_weights: ', agent.critic.model.get_weights())
            print()


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

                rewards_avs.append(np.mean(episode_rewards))

                # reset
                train_reward = running_reward
                episode_ts = []
                episode_rewards = []


                if debug:
                    # plot current population curves
                    plt.figure(figsize = (22.0,12.0))
                    plot_pops(xSol, os.path.join(save_path,'graphs','train','pops_train_' + str(int(episode/test_freq)) + '.png'))
                    np.save(os.path.join(save_path,'data','train','pops_train_' + str(int(episode/test_freq)) + '.npy'), xSol)

            else:
                episode_rewards.append(running_reward)
                episode_ts.append(t)

        # plot results
        plt.figure(figsize = (16.0,12.0))
        plot_survival(time_avs,
                      os.path.join(save_path,'graphs','train_survival.png'),
                      NUM_EPISODES, T_MAX, 'Training')


        plt.figure(figsize = (22.0,12.0))
        plot_pops(xSol, os.path.join(save_path,'graphs','pops.png'))



        plt.figure(figsize = (16.0,12.0))
        plot_rewards(rewards_avs,
                     os.path.join(save_path,'graphs','train_rewards.png'),
                     NUM_EPISODES,T_MAX, 'Training')

        # save results
        np.save(os.path.join(save_path,'data','Pops.npy'), xSol)
        np.save(os.path.join(save_path,'data','Qtrain_rewards.npy'), rewards_avs)
        np.save(os.path.join(save_path,'data','train_survival.npy'), time_avs)


if __name__ == '__main__':
    try:
        r = sys.argv[1]
    except:
        r = 0
    learn(r)
