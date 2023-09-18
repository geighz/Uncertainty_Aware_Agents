#from DQN import *
from DQN_Bayes import *
#from stochastic_gridworld import *
# from two_goalworld import *
from import_game import *#GAME_ENV,state_size
import torch
import torch.optim as optim
from abc import ABC, abstractmethod
import time
from ReplayMemorySingle import ReplayMemorySingle
import copy
# from math import

# since it may take several moves to goal, making gamma high
GAMMA = 0.9
# criterion = torch.nn.MSELoss()#reduction='none'
# criterion_log_like = torch.nn.GaussianNLLLoss()

# def w2_multi_bandidt(target,input,rounds,non_final_mask,times_visited_terminal):  
#     total_error = torch.zeros_like(target)    
#     #Non terminal transitions    
#     if True in non_final_mask:
        
#         total_error[non_final_mask] = torch.abs(target[non_final_mask] -input[non_final_mask])
        
#     #Terminal transitions
#     if False in non_final_mask:
#         mu_approx = times_visited_terminal[:,1]
#         std = times_visited_terminal[:,2]
        
        
#         total_error[~non_final_mask,0]  = torch.abs(input[~non_final_mask,0]-(mu_approx))
#         total_error[~non_final_mask,1] = torch.abs(input[~non_final_mask,1]-std)
        
#         loss_terminal = (torch.mean(input[~non_final_mask,0]),torch.mean(input[~non_final_mask,1]))
#     else: loss_terminal = (0,0)
#     error = torch.sum(torch.linalg.vector_norm(total_error,dim = 0))#/len(target)
#     return error,loss_terminal
def w2(target,input,old_inp,non_final_mask):  
    total_error = torch.zeros_like(target)

    # Error in Q-value prediction 
    total_error[:,0] = (target[:,0]-input[:,0])**2
    # Error of error prediction
    total_error[:,1] =(input[:,1] - (torch.abs(old_inp[:,0]-target[:,0])+ target[:,1]))**2

    
    # #Non terminal transitions    
    # if True in non_final_mask:
    #     total_error[non_final_mask] = torch.abs(target[non_final_mask] -input[non_final_mask])**2
    # #Terminal transitions
    # if False in non_final_mask:
    #     #MU
    #     total_error[~non_final_mask,0]  = torch.abs(target[~non_final_mask,0]-input[~non_final_mask,0])
        
    #     #STD
    #     total_error[~non_final_mask,1]  = torch.abs(input[~non_final_mask,1]-total_error[~non_final_mask,0] )**2
    #     total_error[~non_final_mask,0] =total_error[~non_final_mask,0]**2
        
    
    
    
    error = torch.sum(torch.sum(total_error,dim = 0)/len(target))/2
    return error
   
def hash_state(state):
    if torch.is_tensor(state):
        hash_value = list(state.cpu().numpy().astype(int))
    else:
        hash_value = state.flatten().astype(int)
    hash_value = bin(int(''.join(map(str, hash_value)), 2) << 1)
    return hash_value

class Miner_Single(ABC):
    def __init__(self, number_heads,buffer, budget, va, vg,agent_type_loss, agent_type_train,agent_type_eval,ID):
        #State size = 80.. or 125
        self.state_size = GAME_ENVS[ID][4]#state_size#80#125
        self.reward_history = []
        self.episode_ids = np.array([])
        self.asked_history = np.array([])
        self.adviser_history = np.array([])
        self.memory = ReplayMemorySingle(buffer)
        self.env = GAME_ENVS[ID][0]
        self.ACTION_SPACE_LIST = GAME_ENVS[ID][2]
        self.number_of_eval_games = GAME_ENVS[ID][3]
        self.ID = ID
        self.uncertainty = np.array([])
        self.epsilon = 1
        self.number_heads = number_heads
        self.uncertainty = []
        self.agent_type_loss = agent_type_loss
        self.agent_type_train = agent_type_train
        self.agent_type_eval = agent_type_eval
        self.ACTION_SPACE = GAME_ENVS[ID][1]
        #[164, 150]
        self.policy_net = Bootstrapped_DQN(number_heads, self.state_size, [164, 150], self.ACTION_SPACE, hidden_unit)
        self.target_net = Bootstrapped_DQN(number_heads, self.state_size, [164, 150], self.ACTION_SPACE, hidden_unit)
        self.update_target_net()
        self.times_asked = 0
        self.times_advisee = 0
        self.times_adviser = 0
        self.optimizers = []
        self.measured_loss = []
        self.optim_loss = 0
        self.terminal = {}
        self.budget = budget
        self.rounds = 0
        self.agent_list = []
        self.va = va
        self.vg = vg
        self.times_won = 0
        # self.state_counter = []
        # self.env = GAME_ENV()#Goldmine()#TwoGoal()
        # Fuer jeden head gibt es einen optimizer
        #there is one for every head optimizer
        for head_number in range(self.policy_net.number_heads):
            self.optimizers.append(optim.Adam(self.policy_net.nets[head_number].parameters()))#,lr = 0.0001))
            self.terminal[head_number] = {'ep':[],'mean':[],'std':[]}
            # self.state_counter.append({})
            # optimizers_a.append(optim.SGD(agent_a.model.heads[i].parameters(), lr=0.002))
            # optimizers_b.append(optim.SGD(agent_bQVAL = self.policy_net(states).model.heads[i].parameters(), lr=0.002))

    
    # # Add hashing for visited terminal states
    # def count_state(self, state,head,sample):
    #     hash_of_state = hash_state(state)
    #     if hash_of_state in self.state_counter[head]:
    #         #update number of times visited
    #         self.state_counter[head][hash_of_state]['visited'] += 1
    #         # Do we have to copy here?
    #         previous_mean = self.state_counter[head][hash_of_state]['mean'].clone()
    #         #update mean
    #         self.state_counter[head][hash_of_state]['mean'] += (sample -self.state_counter[head][hash_of_state]['mean'])/self.state_counter[head][hash_of_state]['visited']
    #         #update std
    #         if self.state_counter[head][hash_of_state]['visited'] >= 2:
    #             first_term = (self.state_counter[head][hash_of_state]['visited']-2)*self.state_counter[head][hash_of_state]['std']**2
    #             second_term = (sample -self.state_counter[head][hash_of_state]['mean'] )*(sample - previous_mean)
    #             self.state_counter[head][hash_of_state]['std'] = torch.sqrt((first_term+second_term)/(self.state_counter[head][hash_of_state]['visited']-1))
    #         else: 
    #             self.state_counter[head][hash_of_state]['std'] = 0

    #     else:
    #         self.state_counter[head][hash_of_state] = {'visited': 1, 'mean': sample, 'std': 0 }
    #         # self.state_counter[head][hash_of_state] = 1

    # def times_visited(self, state,head):
    #     hash_of_state = hash_state(state)
    #     if hash_of_state in self.state_counter[head]:
    #         # print([x for x in self.state_counter[head][hash_of_state].values()])
    #         return [x for x in self.state_counter[head][hash_of_state].values()]#self.state_counter[head][hash_of_state].values()
    #     else:
    #         return 0
    
    
    def set_partner(self, other_agent_list):
        self.other_agent_list = copy.deepcopy(other_agent_list)
        self.other_agent_list.pop(self.ID)
    
    def give_advise(self, env):
        prob_give = self.probability_advise_in_state(env)
        if self.times_adviser >= self.budget:
            return None
       
        # inv_state = get_grid_for_player(env.state, np.array([0, 0, 0, 0, 1]))
        # action = self.choose_best_action(env)

        #q_values = self.q_circumflex(env)
        return prob_give

    @abstractmethod
    def probability_advise_in_state(self, state):
        pass
    # Look at SingleAwareMiner
    @abstractmethod
    def probability_ask_in_state(self, env):
        pass
    
    def exploration_strategy(self, v_state, epsilon):
        # choose random action
        if np.random.random() < epsilon:
            #Regular epsion-greedy
            # action = np.random.choice(self.ACTION_SPACE_LIST)#np.random.randint(0, ACTION_SPACE)
            # Thomson heuristic
            action = self.thomson_sampling(v_state=v_state)
            # print("A takes random action {}".format(action_a))a
        else:  # choose best action from Q(s,a) values
            action = self.choose_best_action(v_state)
            # print("A takes best action {}".format(action_a))
        return action

    def thomson_sampling(self,v_state):
        qvals = self.policy_net.q_circumflex(v_state=v_state)
        samples = torch.normal(mean= qvals[0],std=qvals[1])
        action = torch.argmax(samples)
        return action


    # This is choosing an action
    
    def choose_training_action(self, env, epsilon):
    
        action = None
        recieved_advice = False

        if self.times_advisee < self.budget:
            prob_ask = self.probability_ask_in_state(env)
            if prob_ask:
                self.times_asked +=1
                q_vals = self.policy_net.q_circumflex(v_state=env)
                current_means = q_vals[0].clone()
                current_stds = q_vals[1].clone()
                # other_agent_uncertainties = []
                for i, other_agent in enumerate(self.agent_list):
                    give_advise = other_agent.probability_advise_in_state(env)
                    if give_advise:
                        ensemble_qvals = other_agent.policy_net.q_circumflex(v_state=env)
                        other_means,other_stds = ensemble_qvals[0],ensemble_qvals[1]
                        idx = torch.where(other_stds < current_stds)[0]
                        if idx.numel():
                            # print(f'advice to {self.ID} from {other_agent.ID}')
                            # print(self.env)
                            self.times_advisee += 1
                            recieved_advice = True
                            current_means[idx] = other_means[idx].clone()
                            current_stds[idx] = other_stds[idx].clone()                
                if recieved_advice:
                    q_vals[0] = current_means
                    q_vals[1] = current_stds
                    action = self.choose_best_action(qvals = q_vals)
                else:
                    action = self.exploration_strategy(v_state = env,epsilon=self.epsilon)
            else:
                action = self.exploration_strategy(v_state = env,epsilon=self.epsilon)
        
        else:
            action = self.exploration_strategy(v_state = env,epsilon=self.epsilon)

        return action 

                    
                #TODO ask agents for advice

    
        #TODO:  when doing action advising we need to add it here!!!!!!
        # action = self.exploration_strategy(env,epsilon)
    
        return action
    
    def choose_best_action(self,v_state=None, qvals=None,training=True):
        if qvals == None:
            qvals = self.policy_net.q_circumflex(v_state=v_state)

        
        
        all_mu_and_sigs = qvals
    

        if training:       
            if self.agent_type_train == 'S':
                incentive = (1-0.5*(all_mu_and_sigs[1]/(all_mu_and_sigs[1]+1)))
                state_action_values_joint = all_mu_and_sigs[0] * incentive
            elif self.agent_type_train == 'R':
                incentive = (1+0.5*(all_mu_and_sigs[1]/(all_mu_and_sigs[1]+1)))
                state_action_values_joint = all_mu_and_sigs[0] * incentive
            elif self.agent_type_train == 'N':
                state_action_values_joint = all_mu_and_sigs[0]
            else:
                print(' non valid agent ')
                return
            return torch.argmax(state_action_values_joint)
        else:
            if self.agent_type_eval == 'S':
                incentive = (1-0.5*(all_mu_and_sigs[1]/(all_mu_and_sigs[1]+1)))
                state_action_values_joint = all_mu_and_sigs[0] * incentive
            elif self.agent_type_eval == 'R':
                incentive = (1+0.5*(all_mu_and_sigs[1]/(all_mu_and_sigs[1]+1)))
                state_action_values_joint = all_mu_and_sigs[0] * incentive
            elif self.agent_type_eval == 'N':
                state_action_values_joint = all_mu_and_sigs[0]
            else:
                print(' non valid agent ')
                return
            return torch.argmax(state_action_values_joint)

              

    def get_state_action_value_distributions(self, state, action, current=True):
        #qval: for each head retrieve the qvals for the state
        if current == True:
            qval = self.policy_net(state)
        else:
            qval = self.target_net(state)
        # for each head, stack the mean and variance of the action for each batch
        # Don't worry about the squeeze 
        return  [torch.stack((qval_head[0].gather(1, action),qval_head[1].gather(1, action)),-1).squeeze(1)   for qval_head in qval]
    #NOT DONE
    def optimize(self, states, actions, new_states, rewards, non_final_mask):
        # #Now each state has two values, one for for mean and one for  standard deviation.
        
        #Original value: 
        # mean + one std
        
        
        qval_heads = self.target_net(new_states)
        #old_qval_heads = self.target_net(states)
   
        #Obtain the mean for the largest mean+std

        if self.agent_type_loss == 'S':#all_mu_and_sigs[0]*(1-0.5*(all_mu_and_sigs[1]/(all_mu_and_sigs[1]+1))) 
            value_next_state_per_head = [[qval[0][np.arange(len(qval[0])),[(qval[0]*(1-.5*(qval[1]/(qval[1]+1)))).argmax(1)][0]],qval[1][np.arange(len(qval[0])),[(qval[0]*(1-.5*(qval[1]/(qval[1]+1)))).argmax(1)][0]]] for qval in qval_heads]
            
        elif self.agent_type_loss == 'R':
            value_next_state_per_head = [[qval[0][np.arange(len(qval[0])),[(qval[0]*(1+.5*(qval[1]/(qval[1]+1)))).argmax(1)][0]],qval[1][np.arange(len(qval[0])),[(qval[0]*(1+.5*(qval[1]/(qval[1]+1)))).argmax(1)][0]]] for qval in qval_heads]
            
        elif self.agent_type_loss == 'N':
            value_next_state_per_head = [[qval[0][np.arange(len(qval[0])),[(qval[0]).argmax(1)][0]],qval[1][np.arange(len(qval[0])),[(qval[0]).argmax(1)][0]]] for qval in qval_heads]
        else:
            print('non valid agent')
            assert True == False
       
        
        targ_per_head = []
        EPSILON = 0.00001*torch.zeros_like(rewards)
        
        for value_next_state in value_next_state_per_head:
            # Store mean and std 
            # Terminal states have mean equal to reward and std = eps, check paper if in doubt.

            target_mu = rewards.clone()
            target_std = EPSILON.clone()            
            target_mu[non_final_mask] +=  GAMMA*value_next_state[0][non_final_mask]
            target_std[non_final_mask] = GAMMA**(.5) * value_next_state[1][non_final_mask]
            target_mu.detach()
            target_std.detach()
            
            targ_per_head.append( torch.column_stack((target_mu,target_std)))
        #print(f'time value next state {t1 -time.time()}')  
        
        state_action_values = self.get_state_action_value_distributions(states, actions)
        state_action_values_old = self.get_state_action_value_distributions(states, actions,current=False)
        #print(f'time get state act {t1 -time.time()}')  
        
        loss = []
        #Per head, first column is the mean and second column is the standard deviation
        
        # loss_terminal_heads = torch.zeros((5,2))
        for head in range(self.number_heads):
            # size = 10 because the batch is of size 10.
            '''
            use_sample = np.random.randint(self.number_heads, size=10) == 0
            while True not in use_sample:
                use_sample = np.random.randint(self.number_heads, size=10) == 0
            
            non_final_mask_cur_head = non_final_mask[use_sample]

            # used_states = new_states[use_sample]
            # used_rewards = rewards[use_sample]

            # for i,state in enumerate(used_states[~non_final_mask_cur_head]):
            #     self.count_state(state,head=head,sample = used_rewards[~non_final_mask_cur_head][i]) 
            
            inp = state_action_values[head]
            
            
            target = targ_per_head[head]            
            inp =inp[use_sample] 
            target = target[use_sample] 
            
            # times_visited_terminal = torch.tensor([self.times_visited(state,head=head) for state in used_states[~non_final_mask_cur_head]])
            loss_total,loss_terminal = w2(target,inp,self.rounds,non_final_mask=non_final_mask_cur_head)
            '''
            inp = state_action_values[head].clone()#.view(10)
            old_inp = state_action_values_old[head].clone()
                
            target = targ_per_head[head].clone()#.view(10)
            
                
            use_sample = np.random.randint(self.number_heads, size=10) != 0
            # while True not in use_sample and :
            #     use_sample = np.random.randint(self.number_heads, size=10) != 0
            inp[use_sample] *= 0
            target[use_sample] *= 0
            old_inp[use_sample] *= 0
            loss_total = w2(target,inp,old_inp,non_final_mask=non_final_mask)
            # loss_total,loss_terminal = w2_multi_bandidt(target,inp,self.rounds,
            # non_final_mask=non_final_mask_cur_head,times_visited_terminal=times_visited_terminal)
            # loss.append(criterion_mse(inp, target))
            loss.append(loss_total)
            # loss.append(criterion(inp, target))
        self.measured_loss+= torch.sum(loss_total).detach().numpy()
        self.optim_loss+=1

        for head in range(self.number_heads):
            # clear gradient
            self.optimizers[head].zero_grad()
            # loss[head].backward(retain_graph=True)
            loss[head].backward()
            # update model parameters
            self.optimizers[head].step()

    
        #print(f'time optimize {t1 -time.time()}')  
        #count states terminal states for loss function
        
        # return loss_terminal_heads
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
    def calculate_uncertainty(self, v_state,pessimistic=True):
        qval = self.policy_net(v_state)
        actions = len(qval[0][0][0])
        # return 1
        if self.number_heads <2:
            uncertainty_measure = (torch.max(qval[0][1]) if pessimistic else torch.min(qval[0][1])).detach().numpy()
            self.uncertainty.append(uncertainty_measure)
            return uncertainty_measure

        min_action = float('inf')
        max_action = -1
        norm_variance = 0


        for action in range(actions):
            min_heads, _ = torch.min(torch.stack([head[1][0][action] for head in qval]).detach(), dim=0)
            max_heads, _ = torch.max(torch.stack([head[1][0][action] for head in qval]).detach(), dim=0)
            
            # stds_action = torch.stack([head[1][0][action] for head in qval])
            means_action = torch.stack([head[0][0][action] for head in qval])

            min_action = min(min_action, max_heads.item())
            max_action = max(max_action, min_heads.item())

            norm_variance += self.variance(means_action).detach()

            #norm_variance += self.variance(means_actions)

        uncertainty_measure = norm_variance+min_action if pessimistic else norm_variance + max_action
        # var_uncertainty = min_action if pessimistic else max_action
        # uncertainty = var_uncertainty+norm_variance
        self.uncertainty.append(uncertainty_measure.detach().numpy())
        #print(f'time uq computation {t1 - time.time()}')
        return uncertainty_measure.detach().numpy()