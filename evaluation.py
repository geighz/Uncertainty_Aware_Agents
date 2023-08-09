#from stochastic_gridworld import Goldmine
from import_game import *#GAME_ENV,number_of_eval_games
# from two_goalworld import *
import numpy as np
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm 
#number_of_eval_games = 506
# number_of_eval_games = 150


def evaluate_agents(agent_a, agent_b):
    reward_sum = 0
    #Can evaluate on different environments...
    env = GAME_ENV()
    show_heat_map = False
    # dictionaries for quiver and heat map
    if show_heat_map:
        num_of_possible_states = 23
        qval_dictionary_a = dict.fromkeys(list(range(0,num_of_possible_states)), 0)
        quiver_dictionary_a = dict.fromkeys(list(range(0,num_of_possible_states)), np.array([0,0]))
        qval_dictionary_b = dict.fromkeys(list(range(0,num_of_possible_states)), 0)

    #env = TwoGoal()
    agent_a.reset_uncertainty()
    agent_b.reset_uncertainty()
    for state_id in tqdm(range(number_of_eval_games), desc='Evaluating games'):
        
        env.reset(state_id)
        
        # env.render()
        steps = 0
        done = False
        while not done:
            action_a = agent_a.choose_best_action_ev(env.v_state)
            
            agent_a.probability_ask_in_state(env)
            action_b = agent_b.choose_best_action_ev(env.v_state)
            agent_b.probability_ask_in_state(env)
            #

            # Creation of heat map
            if show_heat_map and steps==0:
                # Using floor and module to access correct locations for storing the policy
                qval_dictionary_a[state_id//(num_of_possible_states-1)] += agent_a.policy_net.q_circumflex(env.v_state).squeeze()[action_a.item()].detach().numpy()
                quiver_dictionary_a[state_id//(num_of_possible_states-1)] =quiver_dictionary_a[state_id//(num_of_possible_states-1)]+ get_action(action_a.item())
                #TODO Fix the heat map for agent B
                if state_id//(num_of_possible_states-1) < state_id%(num_of_possible_states):
                    qval_dictionary_b[(state_id%(num_of_possible_states))-1] += agent_b.policy_net.q_circumflex(env.v_state).squeeze()[action_b.item()].detach().numpy()
                else:
                    qval_dictionary_b[(state_id%(num_of_possible_states))] += agent_b.policy_net.q_circumflex(env.v_state).squeeze()[action_b.item()].detach().numpy()
                # if state_id//(num_of_possible_states-1) == 0:
                    #print('Action a is:', action_a)
            state, reward, done, _ = env.step(action_a, action_b)
         
            # env.render()
            
            reward_sum += reward
            
            steps += 1
            if steps > 10:
                
                done = True
    uncertainty_mean = mean(agent_a.get_uncertainty(), agent_b.get_uncertainty())
    if show_heat_map:       
        # Heat plot
        values = list(qval_dictionary_a.values())
        max_ = max(values)+1
        values.insert(4,max_)
        values.insert(20,max_)
        values = np.array(values).reshape(5,5)
        fig = plt.figure()
        fig.set_size_inches((1,1))
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        plt.set_cmap('hot')
        ax.imshow(values, interpolation='sinc')
        plt.savefig('heat.png', dpi=80)

        #TODO: Fix quiver plot
        # quiver_vals = list(quiver_dictionary_a.values())
        # x,y = np.arange(0,5,1),np.arange(0,5,1)
        # X,Y = np.meshgrid(x,y)
        # quiver_vals.insert(4,np.array([0,0]))
        # quiver_vals.insert(20,np.array([0,0]))
        # quiver_vals = np.array(quiver_vals)/(num_of_possible_states-1)
        # norms = np.linalg.norm(quiver_vals,axis = 1)
        # quiver_vals = np.array([quiver_vals[i]/norms[i] if norms[i] > 0 else quiver_vals[i] for i in range(len(norms))])
        # fig,ax = plt.subplots(figsize = (9,9))
        # ax.quiver(X,Y,quiver_vals[:,0],quiver_vals[:,1])
        # plt.show()
        # plt.close()
    return reward_sum / number_of_eval_games, uncertainty_mean


def mean(*vas):
    array = np.array([])
    for va in vas:
        array = np.append(array, np.asarray(va, dtype=np.float32))
    if len(array) > 0:
        return np.average(array, axis=0)
    else:
        return 0
