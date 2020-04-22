from evaluation import *
from Miner import *
from Plotter import *
from ReplayMemory import ReplayMemory

epochs = 1001
number_heads = 4
TARGET_UPDATE = 5
BUFFER = 80
BATCH_SIZE = 10


class Main:
    def __init__(self):
        self.epsilon = 1
        self.agent_a = Miner(number_heads)
        self.agent_b = Miner(number_heads)
        self.agent_a.set_partner(self.agent_b)
        self.agent_b.set_partner(self.agent_a)
        self.reward_history = []
        self.x = []
        self.advisee_history = []
        self.adviser_history = []
        self.memory = ReplayMemory(BUFFER)
        self.env = Goldmine()

    def track_progress(self, episode_number):
        if episode_number % 25 == 0:
            self.x.append(episode_number)
            average_reward = evaluate_agents(self.agent_a, self.agent_b)
            self.reward_history.append(average_reward)
            self.advisee_history.append(self.agent_a.times_advisee)
            self.adviser_history.append(self.agent_a.times_adviser)
            self.agent_a.times_advisee = 0
            self.agent_a.times_adviser = 0
        if episode_number % 1000 == 0 and not episode_number == 0:
            plot(self.x, self.reward_history, self.advisee_history, self.adviser_history)

    def train_and_evaluate_agent(self):
        for i_episode in range(epochs):
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
                if len(self.memory) < BUFFER:
                    if done:
                        break
                    else:
                        continue
                states, actions_a, actions_b, new_states, rewards, non_final = self.memory.sample(BATCH_SIZE)
                self.agent_a.optimize(states, actions_a, new_states, rewards, non_final)
                self.agent_b.optimize(states, actions_b, new_states, rewards, non_final)
                if done:
                    self.track_progress(i_episode)
                if step > 20:
                    break
            if self.epsilon > 0.02:
                self.epsilon -= (1 / epochs)
            if i_episode % TARGET_UPDATE == 0:
                for head_number in range(self.agent_a.policy_net.number_heads):
                    self.agent_a.update_target_net()
                    self.agent_b.update_target_net()
        return self.x, self.reward_history, self.advisee_history, self.adviser_history


m = Main()
m.train_and_evaluate_agent()
