from gridworld import Goldmine
import numpy as np

total_number_of_eval_games = 150


def evaluate_agents(agent_a, agent_b):
    reward_sum = 0
    env = Goldmine()
    agent_a.reset_uncertainty()
    agent_b.reset_uncertainty()
    for state_id in range(number_of_eval_games):
        env.reset(state_id)
        # env.render()
        steps = 0
        done = False
        while not done:
            action_a = agent_a.choose_best_action(env.v_state)
            agent_a.probability_ask_in_state(env)
            action_b = agent_b.choose_best_action(env.v_state)
            agent_b.probability_ask_in_state(env)
            state, reward, done, _ = env.step(action_a, action_b)
            # env.render()
            reward_sum += reward
            steps += 1
            if steps > 10:
                done = True
    uncertainty_mean = mean(agent_a.get_uncertainty(), agent_b.get_uncertainty())
    return reward_sum / number_of_eval_games, uncertainty_mean


def mean(*vas):
    array = np.array([])
    for va in vas:
        array = np.append(array, np.asarray(va, dtype=np.float32))
    if len(array) > 0:
        return np.average(array, axis=0)
    else:
        return 0
