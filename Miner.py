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


def variance(predictions):
    predictions = torch.stack(predictions)
    return predictions.var(dim=0)


class Miner(ABC):
    # TODO: number of heads belongs to the subclass not the parent class
    def __init__(self, number_heads):
        self.number_heads = number_heads
        self.policy_net = Bootstrapped_DQN(number_heads, 80, [164, 150], 4, hidden_unit)
        self.target_net = Bootstrapped_DQN(number_heads, 80, [164, 150], 4, hidden_unit)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
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
        prob_give = self.probability_advise_in_state(env.state)
        if np.random.random() < prob_give:
            return None
        # give advise
        # print("give advise")
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
        prob_ask = self.probability_ask_in_state(env)
        if np.random.random() < prob_ask:
            # ask for advice
            # print("ask for advice")
            self.times_advisee += 1
            action = self.other_agent.give_advise(env)
        if action is None:
            action = self.exploration_strategy(env, epsilon)
        return action

    def choose_best_action(self, v_state):
        qval = self.policy_net(v_state)
        # q-values derived from all heads, compare with
        # Uncertainty-Aware Action Advising for Deep Reinforcement Learning Agents
        # page 5
        final_q_function = qval[0]
        for i in range(self.number_heads - 1):
            final_q_function += qval[i + 1]
        # print(qval)
        # take action with highest Q-value
        final_q_function = final_q_function.data / self.number_heads
        return np.argmax(final_q_function.data)

    def get_state_action_value(self, state, action):
        qval = self.policy_net(state)
        result = []
        for i in range(self.number_heads):
            result.append(qval[i].gather(1, action))
        return result

    def optimize(self, states, actions, new_states, rewards, non_final_mask):
        state_action_values = self.get_state_action_value(states, actions)
        maxQ = self.target_net.q_circumflex(new_states).max(1)[0]
        target = rewards.clone()
        target[non_final_mask] += GAMMA * maxQ[non_final_mask]
        target = target.detach()
        loss = []
        for a in range(self.number_heads):
            use_sample = np.random.randint(0, self.number_heads)
            if use_sample == 0:
                loss.append(criterion(state_action_values[a].view(10), target))
            else:
                loss.append(None)

        # Optimize the model
        # Clear gradients of all optimized torch.Tensor s.
        for a in range(self.number_heads):
            if loss[a] is not None:
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
