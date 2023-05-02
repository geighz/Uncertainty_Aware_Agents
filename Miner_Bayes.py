#from DQN import *
from DQN_Bayes import *
from gridworld import *
import torch
import torch.optim as optim
from abc import ABC, abstractmethod
from time import time 
# from math import

# since it may take several moves to goal, making gamma high
GAMMA = 0.9
#criterion = torch.nn.MSELoss()
criterion = torch.nn.GaussianNLLLoss()


def loss_expected_log_likelihood(target,input):

    mean = input.T[:][0]

    var = input.T[:][1]**2 

    loss = torch.log(var)/2+(target - mean)**2/(2*var)+0.5*torch.log(2*torch.tensor(torch.pi))

    return torch.mean(torch.log(var)/2+(target - mean)**2/(2*var)+0.5*torch.log(2*torch.tensor(torch.pi)))
   



def hash_state(state):
    if torch.is_tensor(state):
        hash_value = list(state.cpu().numpy().astype(int))
    else:
        hash_value = state.flatten().astype(int)
    hash_value = bin(int(''.join(map(str, hash_value)), 2) << 1)
    return hash_value


class Miner_Bayes(ABC):
    def __init__(self, number_heads, budget, va, vg):
        self.number_heads = number_heads
        self.budget = budget
        self.va = va
        self.vg = vg
        self.uncertainty = []
        self.policy_net = Bootstrapped_DQN(number_heads, 80, [164, 150], 4, hidden_unit)
        self.target_net = Bootstrapped_DQN(number_heads, 80, [164, 150], 4, hidden_unit)
        self.update_target_net()
        self.times_asked = 0
        self.times_advisee = 0
        self.times_adviser = 0
        self.optimizers = []
        self.vars = []
        # Fuer jeden head gibt es einen optimizer
        #there is one for every head optimizer
        for head_number in range(self.policy_net.number_heads):
            self.optimizers.append(optim.Adam(self.policy_net.nets[head_number].parameters()))
            # optimizers_a.append(optim.SGD(agent_a.model.heads[i].parameters(), lr=0.002))
            # optimizers_b.append(optim.SGD(agent_bQVAL = self.policy_net(states).model.heads[i].parameters(), lr=0.002))

    # model.load_state_dict(torch.load('/Users/Lukas/repositories/Reinforcement-Learning-Q-learning-Gridworld-Pytorch/graph_output/model_a.pth'))
    # model.eval()
    #GOOD
    def set_partner(self, other_agent):
        self.other_agent = other_agent
    #MAYBE CHANGE
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
    #GOOD
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
    #POSSIBLY CHANGE
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
    #GOOD
    def choose_best_action(self, v_state):
        state_action_values_joint = self.policy_net.q_circumflex(v_state)
        #action_selected = torch.argmax(state_action_values_joint)
        return torch.argmax(state_action_values_joint)
    #NEEDS TO BE CHANGED
    # def get_state_action_value(self, state, action):
    #     #qval: for each head retrieve the qvals for the state
    #     qval = self.policy_net(state)
    #     # Each head has [mean_1, std_1,,mean_2,std_2...]
        
    #     #gathers = [qval_head.gather(1, action) for qval_head in qval]
        
    #     # qval_head gets the q_vals per head, extract means per head with [:,::2] and stds with [:,1::2] 
    #     #return [torch.normal(mean=qval_head[:,::2],std=qval_head[:,1::2]).gather(1,action) for qval_head in qval]
    #     return [torch.normal(mean=qval_head[0],std=qval_head[1]).gather(1,action) for qval_head in qval]
    def get_state_action_value_distributions(self, state, action):
        #qval: for each head retrieve the qvals for the state
        qval = self.policy_net(state)
        # for each head, stack the mean and variance of the action for each batch
        # Don't worry about the squeeze 
        check =1
        return  [torch.stack((qval_head[0].gather(1, action),qval_head[1].gather(1, action)),-1).squeeze(1)    for qval_head in qval]
    #NOT DONE
    def optimize(self, states, actions, new_states, rewards, non_final_mask):
        # #Now each state has two values, one for for mean and one for  standard deviation.
        #gaussian_state_action_values = self.target_net(new_states)
        #Original value: 
        # mean + one std
        qval_heads = self.target_net(new_states)
        #Obtain the mean for the largest mean+std
        value_next_state_per_head = [qval[0][np.arange(len(qval[0])),[(qval[0]+qval[1]).argmax(1)][0]] for qval in qval_heads]
       

        targ_per_head = []

        
        for value_next_state in value_next_state_per_head:
            target = rewards.clone()
            target[non_final_mask] += GAMMA * value_next_state[non_final_mask]
            target = target.detach()
            targ_per_head.append(target)
        state_action_values = self.get_state_action_value_distributions(states, actions)
        
        
        loss = []
        for head in range(self.number_heads):
            
            use_sample = np.random.randint(self.number_heads, size=10) == 0
            while True not in use_sample:
                use_sample = np.random.randint(self.number_heads, size=10) == 0
            # print(state)
            # view reshapes, 10 batches \times 2 (mean,std)
            inp = state_action_values[head]
            # check = targ_per_head[head]
            # num batches \times num_samples 
            target = targ_per_head[head]            
            inp =inp[use_sample] 
            target = target[use_sample] 
            # check = criterion(inp,inp)
            loss.append(loss_expected_log_likelihood(target, inp))
            #loss.append(criterion(target,inp.T[:][0],inp.T[:][1]))
            # loss.append(check)
        # print(loss)

        # Optimize the model
        for head in range(self.number_heads):
            # clear gradient
           
            self.optimizers[head].zero_grad()
            # compute gradients
            # torch.autograd.set_detect_anomaly(True)
            loss[head].backward()
            # update model parameters
            self.optimizers[head].step()
        #print("Loss",loss)
        check = state_action_values[:][:][1]
        

        return torch.var_mean(state_action_values[:][:][1]) 
        #GOOD
    def update_target_net(self):
        for head in range(self.number_heads):
            policy_head = self.policy_net.nets[head]
            target_head = self.target_net.nets[head]
            #COPY OVER WEIGHTS
            target_head.load_state_dict(policy_head.state_dict())
        self.target_net.eval()
    #GOODqval_head[:,::2]
    def get_uncertainty(self):
        return self.uncertainty
    #MAYBE CHANGE
    def reset_uncertainty(self):
        self.uncertainty = []
        
    # def max_gaussian_targets(self, new_states):
    #     # This function is to replace 
    #     # value_next_state_per_head = [x.max(1)[0] for x in self.target_net(new_states)]
    #     #Number of samples for the maximum
    #     number_samples = int(5*10e2)
    #     # First: get values from target net
    #     qval= self.target_net(new_states)


    #     # Per head, obtain main and variance of qvals. 
    #     # Sample from each qval for each badge and get the maximum for each badge
    #     return  [torch.stack([torch.normal(mean=qval_head[:,::2],std=qval_head[:,1::2]).max(1)[0]  for i in range(number_samples)  ]).T  for qval_head in qval]

    # def wasserstein_loss(self,)