from DQN import *
#from gridworld import *
from two_goalworld import *
import torch
import torch.optim as optim
from abc import ABC, abstractmethod
import numpy as np

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
        # 80 for gridworld, 125 for twogoal()
        self.state_size = 125
        self.budget = budget
        self.va = va
        self.vg = vg
        self.uncertainty = []
        self.policy_net = Bootstrapped_DQN(number_heads, self.state_size, [164, 150], 4, hidden_unit)
        self.target_net = Bootstrapped_DQN(number_heads, self.state_size, [164, 150], 4, hidden_unit)
        self.update_target_net()
        self.times_asked = 0
        self.times_advisee = 0
        self.times_adviser = 0
        self.optimizers = []
        # Fuer jeden head gibt es einen optimizer
        #there is one for every head optimizer
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

    def choose_best_action(self, v_state):
        state_action_values_joint = self.policy_net.q_circumflex(v_state)
        return np.argmax(state_action_values_joint.data)

    def get_state_action_value(self, state, action):
        qval = self.policy_net(state)
        return [qval_head.gather(1, action) for qval_head in qval]

    def optimize(self, states, actions, new_states, rewards, non_final_mask):
        check = self.target_net(new_states)
        
        value_next_state_per_head = [x.max(1)[0] for x in self.target_net(new_states)]
        targ_per_head = []
        for value_next_state in value_next_state_per_head:
            target = rewards.clone()
            target[non_final_mask] += GAMMA * value_next_state[non_final_mask]
            target = target.detach()
            targ_per_head.append(target)

        state_action_values = self.get_state_action_value(states, actions)
        loss = []
        loss_terminal_heads = np.zeros((5))
        for head in range(self.number_heads):
            inp = state_action_values[head].view(10)
            if inp[~non_final_mask].numel():
                # print(inp[~non_final_mask])
                loss_terminal_heads[head] = torch.mean(inp[~non_final_mask])
            target = targ_per_head[head].view(10)
            use_sample = np.random.randint(self.number_heads, size=10) != 0
            inp[use_sample] *= 0
            target[use_sample] *= 0
            loss.append(criterion(inp, target))
            

        # Optimize the model
        for head in range(self.number_heads):
            # clear gradient
            self.optimizers[head].zero_grad()
            # compute gradients
            loss[head].backward()
            # update model parameters
            self.optimizers[head].step()
        
        return loss_terminal_heads
    def update_target_net(self):
        for head in range(self.number_heads):
            policy_head = self.policy_net.nets[head]
            target_head = self.target_net.nets[head]
            target_head.load_state_dict(policy_head.state_dict())
        self.target_net.eval()

    def get_uncertainty(self):
        return self.uncertainty

    def reset_uncertainty(self):
        self.uncertainty = []
