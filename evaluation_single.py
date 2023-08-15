#from two_goalworld import *
#from gridworld import *
from import_game import GAME_ENV,number_of_eval_games
import numpy as np
from torch.autograd import Variable
import torch
#number_of_eval_games = 506

# number_of_eval_games = 150

def evaluate_single_agent(agent_a):
    reward_sum = 0
    env = GAME_ENV
    #env = TwoGoal()
    # env = 
    # number_of_eval_games = 5
    agent_a.reset_uncertainty()
    for state_id in range(number_of_eval_games):

        old_v_state, _ = GAME_ENV.reset(seed=state_id)
        old_v_state = Variable(torch.from_numpy(old_v_state)).view(1, -1).detach()
        steps = 0
        done = False

        while not done:

            action_a = agent_a.choose_best_action(old_v_state)
            agent_a.calculate_uncertainty(old_v_state)
            new_v_state, reward, done, _, _ = env.step(action_a)
            reward_sum += reward
            steps += 1

            old_v_state = Variable(torch.from_numpy(new_v_state)).view(1, -1).detach()
            
            if steps > 30:
                done = True

    uncertainty_mean = mean(agent_a.get_uncertainty())
    print(reward_sum)
    print(uncertainty_mean)
    return reward_sum / number_of_eval_games, uncertainty_mean


def mean(*vas):
    array = np.array([])
    for va in vas:
        array = np.append(array, np.asarray(va, dtype=np.float32))
    if len(array) > 0:
        return np.average(array, axis=0)
    else:
        return 0