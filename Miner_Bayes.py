#from DQN import *
from DQN_Bayes import *
#from stochastic_gridworld import *
from two_goalworld import *
import torch
import torch.optim as optim
from abc import ABC, abstractmethod
from time import time 
# from math import

# since it may take several moves to goal, making gamma high
GAMMA = 0.9
criterion_mse = torch.nn.MSELoss()#reduction='none'
criterion_log_like = torch.nn.GaussianNLLLoss()

def w2_multi_bandidt(target,input,rounds,non_final_mask,times_visited_terminal):  
    total_error = torch.zeros_like(target)
    check = input[non_final_mask,0]
    check1 = target[non_final_mask,0]
    
    #Non terminal transitions    
    if True in non_final_mask:
        total_error[non_final_mask,0] = torch.abs(target[non_final_mask,0] -input[non_final_mask,0])
        total_error[non_final_mask,1] = torch.abs( torch.zeros_like(input[non_final_mask,1]) -input[non_final_mask,1])
    #Terminal transitions
    if False in non_final_mask:
        mu_approx = (target[~non_final_mask,0]+times_visited_terminal*input[~non_final_mask,0])*(1/(times_visited_terminal+1))
        check = target[~non_final_mask,0]
        # print(input[~non_final_mask,0],input[~non_final_mask,1])
        # mu_approx = 20*torch.ones_like(input[~non_final_mask,0],dtype= torch.float)
        idx_visited = torch.where(times_visited_terminal>= 2)[0]
        idx_nonvisited = torch.where(times_visited_terminal<2)[0]
        if idx_visited.numel():
            std_error = total_error[:,1].clone()
            std_error_visited = std_error[~non_final_mask].clone()
            first_term = (input[~non_final_mask,:][idx_visited,1]**2)*(times_visited_terminal[idx_visited]-2)
            second_term = (target[~non_final_mask,:][idx_visited,0]-mu_approx[idx_visited])*(target[~non_final_mask,:][idx_visited,0]-input[~non_final_mask,:][idx_visited,0])
            y_std = torch.sqrt((first_term+second_term)/(times_visited_terminal[idx_visited] -1))
            # std_error_visited[idx_visited]= torch.abs(input[~non_final_mask,:][idx_visited,1]-y_std)
            std_error_visited[idx_visited]= torch.abs(input[~non_final_mask,:][idx_visited,1]-torch.zeros_like(input[~non_final_mask,:][idx_visited,1]))
            std_error[~non_final_mask] = std_error_visited
            total_error[:,1] = std_error
            
        if idx_nonvisited.numel():
            total_error[~non_final_mask,:][idx_nonvisited,1]  = target[~non_final_mask,:][idx_nonvisited,1]
        # sig_approx = ((target[~non_final_mask,0]-input[~non_final_mask,0])**2+times_visited_terminal*input[~non_final_mask,1])*(1/(times_visited_terminal+1))
        #MU
        # total_error[~non_final_mask,0]  = torch.abs(input[~non_final_mask,0]-(mu_approx))
        total_error[~non_final_mask,0]  = torch.abs(input[~non_final_mask,0]-target[~non_final_mask,0])
        #STD
        #total_error[~non_final_mask,1]  = torch.abs(input[~non_final_mask,1]-sig_approx )
        # if target[~non_final_mask,0][-1] > 0 and rounds%500==0:
        #     print(target[~non_final_mask,0][-1],input[~non_final_mask,0][-1],input[~non_final_mask,1][-1])
        loss_terminal = (torch.mean(input[~non_final_mask,0]),torch.mean(input[~non_final_mask,1]))
    else: loss_terminal = (0,0)
    error = torch.sum(torch.linalg.vector_norm(total_error,dim = 0))#/len(target)
    return error,loss_terminal

def debug_loss(target,input,rounds,non_final_mask,times_visited_terminal):
    total_error = torch.zeros_like(target)
    
    #Non terminal transitions    
    if True in non_final_mask:
        total_error[non_final_mask] = torch.abs(target[non_final_mask] -input[non_final_mask])
        # print(input[non_final_mask,0])
    #Terminal transitions
    if False in non_final_mask:
        mu_approx_1 = (target[~non_final_mask,0]+times_visited_terminal*input[~non_final_mask,0])*(1/(times_visited_terminal+1))
        print(input[~non_final_mask,0])
        mu_approx = 20*torch.ones_like(input[~non_final_mask,0],dtype= torch.float)
        idx_visited = torch.where(times_visited_terminal>= 2)[0]
        idx_nonvisited = torch.where(times_visited_terminal<2)[0]
        if idx_visited.numel():
            std_error = total_error[:,1].clone()
            std_error_visited = std_error[~non_final_mask].clone()
            first_term = (input[~non_final_mask,:][idx_visited,1]**2)*(times_visited_terminal[idx_visited]-2)
            second_term = (target[~non_final_mask,:][idx_visited,0]-mu_approx_1[idx_visited])*(target[~non_final_mask,:][idx_visited,0]-input[~non_final_mask,:][idx_visited,0])
            y_std = torch.sqrt((first_term+second_term)/(times_visited_terminal[idx_visited] -1))
            std_error_visited[idx_visited]= torch.abs(input[~non_final_mask,:][idx_visited,1]-y_std)
            std_error[~non_final_mask] = std_error_visited
            # total_error[:,1] = std_error
            total_error[:,1] = torch.zeros_like(std_error)
            #print(first_term,second_term)
            #print(input[~non_final_mask,:][idx_visited,1],y_std)
            check = input[~non_final_mask,:][idx_visited,1]
            check2 = torch.abs(input[~non_final_mask,:][idx_visited,1]-y_std)
            check3 = total_error[~non_final_mask,:][idx_visited,1]
            check = 1
        if idx_nonvisited.numel():
            total_error[~non_final_mask,:][idx_nonvisited,1]  = target[~non_final_mask,:][idx_nonvisited,1]
        # sig_approx = ((target[~non_final_mask,0]-input[~non_final_mask,0])**2+times_visited_terminal*input[~non_final_mask,1])*(1/(times_visited_terminal+1))
        #MU
        total_error[~non_final_mask,0]  = torch.abs(input[~non_final_mask,0]-(mu_approx))
        # total_error_t  = torch.abs(20*torch.ones_like(input[~non_final_mask,0])-input[~non_final_mask,0])
        #STD
        #total_error[~non_final_mask,1]  = torch.abs(input[~non_final_mask,1]-sig_approx )
        # if target[~non_final_mask,0][-1] > 0 and rounds%500==0:
        #     print(target[~non_final_mask,0][-1],input[~non_final_mask,0][-1],input[~non_final_mask,1][-1])
        loss_terminal = (torch.mean(input[~non_final_mask,0]),0.*torch.mean(input[~non_final_mask,1]))
        # loss_terminal = (torch.mean(input[~non_final_mask,0]),torch.mean(input[~non_final_mask,1]))
    else: loss_terminal = (0,0)
    error = torch.sum(torch.linalg.vector_norm(total_error,dim = 0))/len(target)
    return error,loss_terminal
def w2(target,input,rounds,non_final_mask):  
    total_error = torch.zeros_like(target)
    # print(input)
    #Non terminal transitions    
    if True in non_final_mask:
        total_error[non_final_mask] = torch.abs(target[non_final_mask] -input[non_final_mask])
    #Terminal transitions
    if False in non_final_mask:
        #MU
        total_error[~non_final_mask,0]  = torch.abs(target[~non_final_mask,0]-input[~non_final_mask,0])
        # total_error_t  = torch.abs(20*torch.ones_like(input[~non_final_mask,0])-input[~non_final_mask,0])
        #STD
        total_error[~non_final_mask,1]  = torch.abs(input[~non_final_mask,1]-total_error[~non_final_mask,0] )
        # if target[~non_final_mask,0][-1] > 0 and rounds%500==0:
        #     print(target[~non_final_mask,0][-1],input[~non_final_mask,0][-1],input[~non_final_mask,1][-1])
        loss_terminal = (torch.mean(input[~non_final_mask,0]),torch.mean(input[~non_final_mask,1]))
    else: loss_terminal = (0,0)
    error = torch.sum(torch.linalg.vector_norm(total_error,dim = 0))/len(target)
    return error,loss_terminal
   
def hash_state(state):
    if torch.is_tensor(state):
        hash_value = list(state.cpu().numpy().astype(int))
    else:
        hash_value = state.flatten().astype(int)
    hash_value = bin(int(''.join(map(str, hash_value)), 2) << 1)
    return hash_value

class Miner_Bayes(ABC):
    def __init__(self, number_heads, budget, va, vg):
        #State size = 80..
        self.state_size = 125
        self.number_heads = number_heads
        self.budget = budget
        self.va = va
        self.vg = vg
        self.uncertainty = []
        self.safe = True
        #[164, 150]
        self.policy_net = Bootstrapped_DQN(number_heads, self.state_size, [164, 150], 4, hidden_unit)
        self.target_net = Bootstrapped_DQN(number_heads, self.state_size, [164, 150], 4, hidden_unit)
        self.update_target_net()
        self.times_asked = 0
        self.times_advisee = 0
        self.times_adviser = 0
        self.optimizers = []
        self.vars = []
        self.rounds = 0
        self.state_counter = []
        self.env_2 = TwoGoal()
        # Fuer jeden head gibt es einen optimizer
        #there is one for every head optimizer
        for head_number in range(self.policy_net.number_heads):
            self.optimizers.append(optim.Adam(self.policy_net.nets[head_number].parameters(), lr=0.0005,weight_decay=1e-4))
            self.state_counter.append({})
            # optimizers_a.append(optim.SGD(agent_a.model.heads[i].parameters(), lr=0.002))
            # optimizers_b.append(optim.SGD(agent_bQVAL = self.policy_net(states).model.heads[i].parameters(), lr=0.002))

    
    # Add hashing for visited terminal states
    def count_state(self, state,head):
        hash_of_state = hash_state(state)
        if hash_of_state in self.state_counter[head]:
            self.state_counter[head][hash_of_state] += 1
        else:
            self.state_counter[head][hash_of_state] = 1

    def times_visited(self, state,head):
        hash_of_state = hash_state(state)
        if hash_of_state in self.state_counter[head]:
            return self.state_counter[head][hash_of_state]
        else:
            return 0
    #
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
        return torch.argmax(state_action_values_joint)

    def get_state_action_value_distributions(self, state, action):
        #qval: for each head retrieve the qvals for the state
        qval = self.policy_net(state)
        # for each head, stack the mean and variance of the action for each batch
        # Don't worry about the squeeze 
        return  [torch.stack((qval_head[0].gather(1, action),qval_head[1].gather(1, action)),-1).squeeze(1)    for qval_head in qval]
    #NOT DONE
    def optimize(self, states, actions, new_states, rewards, non_final_mask):
        # #Now each state has two values, one for for mean and one for  standard deviation.
        #gaussian_state_action_values = self.target_net(new_states)
        #Original value: 
        # mean + one std
        
        
        qval_heads = self.target_net(new_states)
        
        #Obtain the mean for the largest mean+std
        # value_next_state_per_head = [[qval[0][np.arange(len(qval[0])),[(qval[0]+qval[1]).argmax(1)][0]],qval[1][np.arange(len(qval[0])),[(qval[0]+qval[1]).argmax(1)][0]]] for qval in qval_heads]
        if self.safe:
            # value_next_state_per_head = [[qval[0][np.arange(len(qval[0])),[(qval[0]-qval[1]).argmax(1)][0]],qval[1][np.arange(len(qval[0])),[(qval[0]-qval[1]).argmax(1)][0]]] for qval in qval_heads]
            value_next_state_per_head = [[qval[0][np.arange(len(qval[0])),[(qval[0]).argmax(1)][0]],qval[1][np.arange(len(qval[0])),[(qval[0]).argmax(1)][0]]] for qval in qval_heads]
            # value_next_state_per_head = [x[0,:].max(1)[0] for x in qval_heads]
        else:
            value_next_state_per_head = [[qval[0][np.arange(len(qval[0])),[(qval[0]+qval[1]).argmax(1)][0]],qval[1][np.arange(len(qval[0])),[(qval[0]+qval[1]).argmax(1)][0]]] for qval in qval_heads]
       

        targ_per_head = []
        EPSILON = 0.00001*torch.ones_like(rewards)
        
        for value_next_state in value_next_state_per_head:
            # Store mean and std 
            # Terminal states have mean equal to reward and std = eps, check paper if in doubt.

            target_mu = rewards.clone()
            target_std = EPSILON.clone()

            #if stochastic: TODO
            
            target_mu[non_final_mask] +=  GAMMA*value_next_state[0][non_final_mask]
            
            target_std[non_final_mask] = torch.maximum(EPSILON[non_final_mask], GAMMA**(.5) * value_next_state[1][non_final_mask])
            target_mu.detach()
            target_std.detach()
            # if target_mu[non_final_mask].max() > 10:
            #     prev_state = states[non_final_mask][target_mu[non_final_mask].argmax()]
            #     problematic_state  = new_states[non_final_mask][target_mu[non_final_mask].argmax()]
            #     self.env_2.render(prev_state)
            #     self.env_2.render(problematic_state)
            #     print(rewards[non_final_mask])
            #     print(target_mu[non_final_mask])
            #     check = target_mu[non_final_mask]
            targ_per_head.append( torch.column_stack((target_mu,target_std)))
            
        state_action_values = self.get_state_action_value_distributions(states, actions)
        
        
        loss = []
        #Per head, first column is the mean and second column is the standard deviation
        loss_terminal_heads = torch.zeros((5,2))
        for head in range(self.number_heads):
            
            use_sample = np.random.randint(self.number_heads, size=10) == 0
            while True not in use_sample:
                use_sample = np.random.randint(self.number_heads, size=10) == 0
            
            non_final_mask_cur_head = non_final_mask[use_sample]

            used_states = new_states[use_sample]

            for state in used_states[~non_final_mask_cur_head]:
                self.count_state(state,head=head) 
            # print('Head',head,'Dict',self.state_counter[head].values())
            
            
            
            # view reshapes, 10 batches \times 2 (mean,std)
            inp = state_action_values[head]
            # if inp[non_final_mask,0].max() > 1:
            #     prob_index = inp[non_final_mask,0].argmax()
            #     prev_state = states[non_final_mask][prob_index ]
            #     problematic_state  = new_states[non_final_mask][prob_index ]
            #     prob_action = actions[non_final_mask][prob_index ]
            #     self.env_2.render_state(prev_state)
            #     self.env_2.render_state(problematic_state)
            #     print(rewards[non_final_mask])
            #     print(inp[non_final_mask,0])
            #     print(targ_per_head[head][non_final_mask])
            #     print(prob_action)
            #     taco = self.policy_net(prev_state)
            #     check = inp[non_final_mask,0]
            # check = targ_per_head[head]
            # num batches \times num_samples 
            target = targ_per_head[head]            
            inp =inp[use_sample] 
            target = target[use_sample] 
            times_visited_terminal = torch.tensor([self.times_visited(state,head=head) for state in used_states[~non_final_mask_cur_head]])
            # check = criterion(inp,inp)
            #loss.append(loss_expected_log_likelihood(target, inp))
            #loss.append(criterion_mse(target,inp.T[:][0],inp.T[:][1]))
            #loss.append()
            # loss_total = loss.append(criterion_mse(target,inp))
            # loss_total,loss_terminal = w2(target,inp,self.rounds,non_final_mask=non_final_mask_cur_head)
            loss_total,loss_terminal = w2_multi_bandidt(target,inp,self.rounds,
            non_final_mask=non_final_mask_cur_head,times_visited_terminal=times_visited_terminal)

            # loss_total,loss_terminal = debug_loss(target,inp,self.rounds,
            # non_final_mask=non_final_mask_cur_head,times_visited_terminal=times_visited_terminal)
            loss.append(loss_total)
            loss_terminal_heads[head,0], loss_terminal_heads[head,1] = loss_terminal[0],loss_terminal[1]
            check = 1
            # loss.append(check)
        # print(loss)
        self.rounds+=1
        # print('optimize')
        # print(loss_total)
        # Optimize the model
        for head in range(self.number_heads):
            # clear gradient
            self.optimizers[head].zero_grad()
            loss[head].backward(retain_graph=True)
            # update model parameters
            self.optimizers[head].step()
    
        #count states terminal states for loss function
        
        return loss_terminal_heads
        #GOOD
    def update_target_net(self):
        for head in range(self.number_heads):
            policy_head = self.policy_net.nets[head]
            target_head = self.target_net.nets[head]
            #COPY OVER WEIGHTS
            target_head.load_state_dict(policy_head.state_dict())
        self.target_net.eval()

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