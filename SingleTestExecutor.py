
#from evaluation_gold import *
#from import_game import GAME_ENV
from collections import namedtuple
import numpy as np
import os
import torch
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import os.path
import csv
import seaborn as sns
sns.set_theme()
from torch.autograd import Variable
from evaluation_single import evaluate_single_agent


Test_result = namedtuple('Test_result',
                         ('AgentType', 'EPOCH_ID', 'REWARDS',  'UNCERTAINTY','TERMINAL_TRACK_A','Loss','Train','Evaluation'))
Test_setup_single = namedtuple('Test_setup',
                        ('AgentType', 'NUMBER_HEADS', 'EPOCHS', 'BUFFER', 'BATCH_SIZE', 'TARGET_UPDATE','BUDGET','VA',
                        'VG','Loss','Train','Evaluation','Number_Agents'))

class TestExecutor:
    def __init__(self, number_heads, buffer, agent,budget,va,vg,agent_type_loss, agent_type_train,agent_type_eval,number_agents):
        np.random.seed()
        torch.random.seed()
        # self.agent_a = agent(number_heads,buffer,agent_type_loss, agent_type_train,agent_type_eval,ID = 'agent_a')
        # self.agent_b = agent(number_heads,buffer,agent_type_loss, agent_type_train,agent_type_eval,ID = 'agent_b')
        self.agent_list = []
        for i in range(number_agents):
            self.agent_list.append(agent(number_heads,buffer,budget,va,vg,agent_type_loss, agent_type_train,agent_type_eval,i))
        for i in range(number_agents):
           self.agent_list[i].agent_list = self.agent_list

        # self.agent_a.set_partner(self.agent_b)
        # self.agent_b.set_partner(self.agent_a)        
        self.plot = True
        # Render env
        #self.env.render()

    def track_progress(self, episode_number,agent):
        if episode_number % 500 == 0 :
            agent.episode_ids = np.append(agent.episode_ids, episode_number)          
            mean_reward, uncertainty_mean = evaluate_single_agent(agent)
            
            agent.reward_history = np.append(agent.reward_history, mean_reward)
            # agent.uncertainty = np.append(agent.uncertainty, uncertainty_mean)
            self.vars = []
    def train_and_evaluate_agent_episode(self,agent,i_episode,epochs,target_update,batch_size):
        
        # tracking_time = 100      
            
        old_v_state, _ = agent.env.reset()
        # print(agent.env)
        # print(old_v_state)
       
        old_v_state = Variable(torch.from_numpy(old_v_state)).view(1, -1).detach()
        # print(old_v_state)
        done = False
        step = 0
        done_times = 0
        while not done:
            #old_v_state = self.env.gen_obs()
            
            # print(self.env.v_state)
            # t1 = time.time()
            action = agent.choose_training_action(old_v_state, agent.epsilon)
            #action_b = agent_b.choose_training_action(self.env, self.epsilon)
            # Take action, observe new state S'
            new_v_state, reward, done, _, _= agent.env.step(action)
            
            step += 1
            new_v_state = Variable(torch.from_numpy(new_v_state)).view(1, -1).detach()
            # print(f'time training action computation {t1 - time.time()}')
            agent.memory.push(old_v_state, action,  new_v_state, reward, not done)
            old_v_state = new_v_state 

            # if reward > 0:
                
            #     self.track_terminal(agent.number_heads,agent.terminal,old_v_state,action,i_episode)
            
            if done and reward > 0:
                #done_times+=1
                #agent.reward_history.append(reward)
                agent.times_won +=1
                # print(done)
            # if buffer not filled, add to it
            if len(agent.memory) < agent.memory.capacity:
                if done:
                    break
                else:
                    continue
            
            states, actions, new_states, reward, non_final = agent.memory.sample(batch_size)              
            agent.optimize(states, actions, new_states, reward, non_final)
            # print(f'time optimize computation {t1 - time.time()}')
            if step > 30:
                break
        if i_episode%100 == 0 and i_episode > 0:
            print(agent.ID)
            print('Times won:',agent.times_won/100)
            agent.times_won = 0
        
        # if (i_episode%5000 == 0 and i_episode>0 ):
        #     self.plot_terminal_state(i_episode,agent_a_terminal,agent_b_terminal)
    
        if agent.epsilon > 0.02:
            agent.epsilon -= (1 / epochs)#**(1/2)
        
        if i_episode % target_update == 0:
            
            for head_number in range(agent.policy_net.number_heads):
                agent.update_target_net()
        
    def train_and_evaluate_agent(self, epochs, target_update, batch_size):        
        tracking_time = 500
        done_times = 0
        for i_episode in tqdm(range(epochs+1),desc='Training'):   
            for agent in self.agent_list:
                self.track_progress(i_episode,agent)
                
                if i_episode % tracking_time == 0:
                    print("%s Game #: %s, %f,%s,%f,%f,%f" % (os.getpid(), i_episode,agent.reward_history[-1],done_times,agent.times_asked,agent.times_advisee,agent.epsilon))


                self.train_and_evaluate_agent_episode(agent,i_episode,epochs,target_update,batch_size)
        
        # self.plot_terminal_state(i_episode,agent_a_terminal)
        test_result = []
        for agent in self.agent_list:
            agentType = 'PNN-DQN-'+agent.agent_type_loss+agent.agent_type_train+agent.agent_type_eval
            test_result.append(Test_result(agentType, agent.episode_ids, agent.reward_history, agent.uncertainty,agent.terminal,agent.agent_type_loss,agent.agent_type_train,agent.agent_type_eval))
        # print(test_result)
        return test_result
    def track_terminal(self,number_heads,agent_a_terminal,old_v_state,action_a,i_episode):
        for i in range(number_heads):
            agent_a_terminal[i]['ep'].append(i_episode)
            agent_a_terminal[i]['mean'].append(self.agent_a.policy_net(old_v_state.data)[i][0][0][action_a].detach().numpy())
            agent_a_terminal[i]['std'].append(self.agent_a.policy_net(old_v_state.data)[i][1][0][action_a].detach().numpy())
           
        return 
    def plot_terminal_state(self,i_episode,agent_a_terminal):
        if self.agent_a.number_heads == 1:
            i = 0
            plt.plot(agent_a_terminal[i]['ep'],agent_a_terminal[i]['mean'],'o-',label = f'Agent head {i} mean')
            plt.plot(agent_a_terminal[i]['ep'],agent_a_terminal[i]['std'],'*-',label = f'Agent head {i} std')
            plt.legend()
            plt.savefig(f'plots/mean_std_agent_episode_{i_episode}_{self.agent_a.agent_type_loss}{self.agent_a.agent_type_train}{self.agent_a.agent_type_eval}.png')
            plt.close()
        else:
            figa,axs_a = plt.subplots(1,self.agent_a.number_heads,figsize = (15,15))
            
            for i in range(self.agent_a.number_heads):
                axs_a[i].plot(agent_a_terminal[i]['ep'],agent_a_terminal[i]['mean'],'o-',label = f'Agent head {i} mean')
                axs_a[i].plot(agent_a_terminal[i]['ep'],agent_a_terminal[i]['std'],'o-',label = f'Agent  head {i} std')
                axs_a[i].legend()
            plt.savefig(f'plots/mean_std_agent_a_episode_{i_episode}_{self.agent_a.agent_type_loss}{self.agent_a.agent_type_train}{self.agent_a.agent_type_eval}.png')
            
            plt.close(figa)
            
        return




def execute_test_single(test_id, test, return_dict):
    print(test)
    agenttype, number_heads, epochs, buffer, batch_size, target_update,budget,va,vg,agent_type_loss, agent_type_train,agent_type_eval,number_agents= test
    print("test #: %s" % test_id)
    executor = TestExecutor(number_heads, buffer, agenttype,budget,va,vg,agent_type_loss, agent_type_train,agent_type_eval,number_agents)
    return_dict[test_id] = executor.train_and_evaluate_agent(epochs, target_update, batch_size)


def mean(*vas):
    array = np.array([])
    for va in vas:
        array = np.append(array, np.asarray(va, dtype=np.float32))
    if len(array) > 0:
        return np.average(array, axis=0)
    else:
        return 0