# Here is the test of AI
from gridworld import Goldmine
from torch.autograd import Variable
import torch

total_number_of_eval_games = 10


def evaluate_agents(agent_a, agent_b):
    reward_sum = 0
    env = Goldmine()
    for state_id in range(total_number_of_eval_games):
        env.reset(state_id)
        # env.render()
        steps = 0
        done = False
        while not done:
            action_a = agent_a.choose_best_action(env.v_state)
            action_b = agent_b.choose_best_action(env.v_state)
            state, reward, done, _ = env.step(action_a, action_b)
            # env.render()
            reward_sum += reward
            steps += 1
            if steps > 10:
                done = True
    return reward_sum / total_number_of_eval_games
