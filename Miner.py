from DQN import *
from gridworld import *
import torch
import torch.optim as optim
from abc import ABC, abstractmethod


# since it may take several moves to goal, making gamma high
GAMMA = 0.9
criterion = torch.nn.MSELoss()


def hash_state(state):
    if torch.is_tensor(state):
        hash_value = list(state.cpu().numpy().astype(int))
    else:
        hash_value = state.flatten().astype(int)
    hash_value = bin(int(''.join(map(str, hash_value)), 2) << 1)
    return hash_value


class Miner(ABC):
    def __init__(self, number_heads, budget, va, vg):
        self.number_heads = number_heads
        self.budget = budget
        self.va = va
        self.vg = vg
        self.va_history = []
        self.vg_history = []
        self.policy_net = Bootstrapped_DQN(number_heads, 80, [164, 150], 4, hidden_unit)
        self.target_net = Bootstrapped_DQN(number_heads, 80, [164, 150], 4, hidden_unit)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.times_asked = 0
        self.times_advisee = 0
        self.times_adviser = 0
        self.optimizers = []
        # Fuer jeden head gibt es einen optimizer
        for head_number in range(self.policy_net.number_heads):
            self.optimizers.append(optim.Adam(self.policy_net.nets[head_number].parameters()))
            # optimizers_a.append(optim.SGD(agent_a.model.heads[i].parameters(), lr=0.002))
            # optimizers_b.append(optim.SGD(agent_b.model.heads[i].parameters(), lr=0.002))

    # model.load_state_dict(torch.load('/Users/Lukas/repositories/Reinforcement-Learning-Q-learning-Gridworld-Pytorch/graph_output/model_a.pth'))
    # model.eval()

    def set_partner(self, other_agent):
        self.other_agent = other_agent

    def give_advise(self, env):
        if self.times_adviser >= self.budget:
            return None
        prob_give = self.probability_advise_in_state(env.state)
        if np.random.random() > prob_give:
            return None
        self.times_adviser += 1
        inv_state = get_grid_for_player(env.state, np.array([0, 0, 0, 0, 1]))
        action = self.choose_best_action(v_state(inv_state))
        return action

    @abstractmethod
    def probability_advise_in_state(self, state):
        pass

    @abstractmethod
    def probability_ask_in_state(self, env):
        pass

    def exploration_strategy(self, env, epsilon):
        # choose random action
        if np.random.random() < epsilon:
            action = np.random.randint(0, 4)
            # print("A takes random action {}".format(action_a))
        else:  # choose best action from Q(s,a) values
            action = self.choose_best_action(env.v_state)
            # print("A takes best action {}".format(action_a))
        return action

    # This is choosing an action
    def choose_training_action(self, env, epsilon):
        action = None
        if self.times_advisee < self.budget:
            prob_ask = self.probability_ask_in_state(env)
            if np.random.random() < prob_ask:
                self.times_asked += 1
                action = self.other_agent.give_advise(env)
        if action is None:
            action = self.exploration_strategy(env, epsilon)
        else:
            self.times_advisee += 1
        return action

    def choose_best_action(self, v_state, net=None):
        if net is None:
            net = self.policy_net
        final_q_function = net.q_circumflex(v_state)
        return np.argmax(final_q_function.data)

    def get_state_action_value(self, state, action, net=None):
        if net is None:
            net = net
        qval = self.policy_net(state)
        result = []
        for i in range(self.number_heads):
            result.append(qval[i].gather(1, action))
        return result

    def optimize(self, states, actions, new_states, rewards, non_final_mask):
        state_action_values = self.get_state_action_value(states, actions)
        # TODO: Importantly, each one of these value function function heads Q (s,a;θ) is trained against its own
        #  target network Q (s,a;θ−).
        maxQ = self.target_net.q_circumflex(new_states).max(1)[0]
        target = rewards.clone()
        target[non_final_mask] += GAMMA * maxQ[non_final_mask]
        target = target.detach()
        loss = []
        for a in range(self.number_heads):
            # TODO: we only decide whether to use the entire batch not each sample separately
            #  compare to "Uncertainty-Aware Action Advising for Deep Reinforcement Learning Agents":
            #  implementation friendly description
            use_sample = np.random.randint(0, self.number_heads)
            if use_sample == 0:
                loss.append(criterion(state_action_values[a].view(10), target))

        # Optimize the model
        for a in range(len(loss)):
            # clear gradient
            self.optimizers[a].zero_grad()
            # compute gradients
            loss[a].backward()
            # update model parameters
            self.optimizers[a].step()

    def update_target_net(self):
        for head_number in range(self.number_heads):
            policy_head = self.policy_net.nets[head_number]
            target_head = self.target_net.nets[head_number]
            target_head.load_state_dict(policy_head.state_dict())

    def get_va(self):
        va_history = self.va_history
        self.va_history = []
        return va_history

    def get_vg(self):
        vg_history = self.vg_history
        self.vg_history = []
        return vg_history

