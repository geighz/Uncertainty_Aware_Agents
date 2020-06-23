from ReplayMemory import ReplayMemory
from evaluation import *
from collections import namedtuple
import numpy as np


class TestExecutor:
    def __init__(self, number_heads, buffer, agent):
        self.epsilon = 1
        self.agent_b = agent(number_heads)
        self.agent_a = agent(number_heads)
        self.agent_a.set_partner(self.agent_b)
        self.agent_b.set_partner(self.agent_a)
        self.reward_history = []
        self.episode_ids = np.array([])
        self.advisee_history = np.array([])
        self.adviser_history = np.array([])
        self.memory = ReplayMemory(buffer)
        self.env = Goldmine()

    def track_progress(self, episode_number):
        if episode_number % 250 == 0:
            self.episode_ids = np.append(self.episode_ids, episode_number)
            average_reward = evaluate_agents(self.agent_a, self.agent_b)
            self.reward_history = np.append(self.reward_history, average_reward)
            self.advisee_history = np.append(self.advisee_history, self.agent_a.times_advisee)
            self.adviser_history = np.append(self.adviser_history, self.agent_a.times_adviser)
            self.agent_a.times_advisee = 0
            self.agent_a.times_adviser = 0
        # if episode_number % 1000 == 0 and not episode_number == 0:
        #     plot(self.x, self.reward_history, self.advisee_history, self.adviser_history)

    def train_and_evaluate_agent(self, epochs, target_update, batch_size):
        for i_episode in range(epochs):
            self.track_progress(i_episode)
            if i_episode % 50 == 0:
                print("Game #: %s" % (i_episode,))
            self.env.reset()
            done = False
            step = 0
            # while game still in progress
            while not done:
                old_v_state = self.env.v_state
                action_a = self.agent_a.choose_training_action(self.env, self.epsilon)
                action_b = self.agent_b.choose_training_action(self.env, self.epsilon)
                # Take action, observe new state S'
                _, reward, done, _ = self.env.step(action_a, action_b)
                step += 1
                self.memory.push(old_v_state.data, action_a, action_b, self.env.v_state.data, reward, not done)
                # if buffer not filled, add to it
                if len(self.memory) < self.memory.capacity:
                    if done:
                        break
                    else:
                        continue
                states, actions_a, actions_b, new_states, reward, non_final = self.memory.sample(batch_size)
                self.agent_a.optimize(states, actions_a, new_states, reward, non_final)
                self.agent_b.optimize(states, actions_b, new_states, reward, non_final)
                if step > 20:
                    break
            if self.epsilon > 0.02:
                self.epsilon -= (1 / epochs)
            if i_episode % target_update == 0:
                for head_number in range(self.agent_a.policy_net.number_heads):
                    self.agent_a.update_target_net()
                    self.agent_b.update_target_net()
        return self.episode_ids, self.reward_history, self.advisee_history, self.adviser_history


Test_result = namedtuple('Test_result',
                         ('AgentType', 'EPOCH_ID', 'REWARDS', 'TIMES_ADVISEE', 'TIMES_ADVISER'))
Test_setup = namedtuple('Test_setup',
                        ('AgentType', 'NUMBER_HEADS', 'EPOCHS', 'BUFFER', 'BATCH_SIZE', 'TARGET_UPDATE'))


def execute_test(test_id, test, return_dict):
    print(test)
    agenttype, number_heads, epochs, buffer, batch_size, target_update = test
    # TODO rename test_number
    print("test #: %s" % test_id)
    executor = TestExecutor(number_heads, buffer, agenttype)
    episode_ids, reward_history, advisee_history, adviser_history = executor.train_and_evaluate_agent(epochs, target_update,
                                                                                     batch_size)
    test_result = Test_result(agenttype.__name__, episode_ids, reward_history, advisee_history, adviser_history)
    return_dict[test_id] = test_result
