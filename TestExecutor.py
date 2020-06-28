from ReplayMemory import ReplayMemory
from evaluation import *
from collections import namedtuple
import numpy as np
import os


class TestExecutor:
    def __init__(self, number_heads, buffer, agent, budget):
        self.epsilon = 1
        self.agent_b = agent(number_heads, budget)
        self.agent_a = agent(number_heads, budget)
        self.agent_a.set_partner(self.agent_b)
        self.agent_b.set_partner(self.agent_a)
        self.reward_history = []
        self.episode_ids = np.array([])
        self.asked_history = np.array([])
        self.adviser_history = np.array([])
        self.memory = ReplayMemory(buffer)
        self.env = Goldmine()

    def track_progress(self, episode_number):
        if episode_number % 250 == 0:
            self.episode_ids = np.append(self.episode_ids, episode_number)
            average_reward = evaluate_agents(self.agent_a, self.agent_b)
            self.reward_history = np.append(self.reward_history, average_reward)
            times_asked = (self.agent_a.times_asked + self.agent_b.times_asked) / 2
            self.asked_history = np.append(self.asked_history, times_asked)
            times_adviser = (self.agent_a.times_adviser + self.agent_b.times_adviser) / 2
            self.adviser_history = np.append(self.adviser_history, times_adviser)

    def train_and_evaluate_agent(self, epochs, target_update, batch_size):
        for i_episode in range(epochs):
            self.track_progress(i_episode)
            if i_episode % 250 == 0:
                print("%s Game #: %s" % (os.getpid(), i_episode))
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
        test_result = Test_result(type(self.agent_a).__name__ + self.agent_a.number_heads, self.episode_ids, self.reward_history, self.asked_history, self.adviser_history)
        return test_result


Test_result = namedtuple('Test_result',
                         ('AgentType', 'EPOCH_ID', 'REWARDS', 'TIMES_ASKED', 'TIMES_GIVEN'))
Test_setup = namedtuple('Test_setup',
                        ('AgentType', 'NUMBER_HEADS', 'EPOCHS', 'BUFFER', 'BATCH_SIZE', 'TARGET_UPDATE', 'BUDGET'))


def execute_test(test_id, test, return_dict):
    print(test)
    agenttype, number_heads, epochs, buffer, batch_size, target_update, budget = test
    # TODO rename test_number
    print("test #: %s" % test_id)
    executor = TestExecutor(number_heads, buffer, agenttype, budget)
    return_dict[test_id] = executor.train_and_evaluate_agent(epochs, target_update, batch_size)