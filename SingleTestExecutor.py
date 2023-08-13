from ReplayMemory import ReplayMemory
#from evaluation_gold import *
from import_game import GAME_ENV
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

class TestExecutor:
    def __init__(self, number_heads, buffer, agent, budget, va, vg,agent_type_loss, agent_type_train,agent_type_eval):
        np.random.seed()
        torch.random.seed()
        self.epsilon = 1
        self.agent_a = agent(number_heads, budget, va, vg,agent_type_loss, agent_type_train,agent_type_eval)
        self.reward_history = []
        self.episode_ids = np.array([])
        self.asked_history = np.array([])
        self.adviser_history = np.array([])
        self.memory = ReplayMemory(buffer)
        self.env = GAME_ENV
        self.uncertainty = np.array([])
        self.plot = True
        # Render env
        #self.env.render()

    def track_progress(self, episode_number):
        if episode_number % 100 == 0 and episode_number >0:
            self.episode_ids = np.append(self.episode_ids, episode_number)
            agent_a = self.agent_a
          
            mean_reward, uncertainty_mean = evaluate_single_agent(agent_a)
            self.reward_history = np.append(self.reward_history, mean_reward)
            
            
            self.uncertainty = np.append(self.uncertainty, uncertainty_mean)
            # self.vars = []

    def train_and_evaluate_agent(self, epochs, target_update, batch_size):
        agent_a_terminal = {}
 
        for i in range(self.agent_a.number_heads):
            agent_a_terminal[i] = {'ep':[],'mean':[],'std':[]}
        
        
        
        done_times = 0
        for i_episode in tqdm(range(epochs),desc='Training'):
            
            tracking_time = 100
            if i_episode % tracking_time == 0 and i_episode >0:
                self.track_progress(i_episode)
                print("%s Game #: %s, %f, %f,%s" % (os.getpid(), i_episode,self.reward_history[-1],self.uncertainty[-1],done_times))
                
            old_v_state, _ = self.env.reset()
            old_v_state = Variable(torch.from_numpy(old_v_state)).view(1, -1).detach()
            done = False
            step = 0

            while not done:
                #old_v_state = self.env.gen_obs()
               
                # print(self.env.v_state)
                # t1 = time.time()
                action_a = self.agent_a.choose_training_action(old_v_state, self.epsilon)
                #action_b = self.agent_b.choose_training_action(self.env, self.epsilon)
                # Take action, observe new state S'
                new_v_state, reward, done, _, _= self.env.step(action_a)
                step += 1
                new_v_state = Variable(torch.from_numpy(new_v_state)).view(1, -1).detach()
                # print(f'time training action computation {t1 - time.time()}')
                self.memory.push(old_v_state, action_a, 0,  new_v_state, reward, not done)
                old_v_state = new_v_state 

                #if reward > 0:
                   
                    #self.track_terminal(self.agent_a.number_heads,agent_a_terminal,old_v_state,action_a,i_episode)
                
                if done:
                    done_times+=1
                # if buffer not filled, add to it
                if len(self.memory) < self.memory.capacity:
                    if done:
                        break
                    else:
                        continue
                
                states, actions_a, actions_b, new_states, reward, non_final = self.memory.sample(batch_size)              
                self.agent_a.optimize(states, actions_a, new_states, reward, non_final)
                # print(f'time optimize computation {t1 - time.time()}')
                if step > 30:
                    break
            

            # if (i_episode%5000 == 0 and i_episode>0 ):
            #     self.plot_terminal_state(i_episode,agent_a_terminal,agent_b_terminal)
        
            if self.epsilon > 0.02:
                self.epsilon -= (1 / epochs)
            
            if i_episode % target_update == 0:
                
                for head_number in range(self.agent_a.policy_net.number_heads):
                    self.agent_a.update_target_net()
        
        # self.plot_terminal_state(i_episode,agent_a_terminal,agent_b_terminal)
        agentType = 'PNN-DQN-'+self.agent_a.agent_type_loss+self.agent_a.agent_type_train+self.agent_a.agent_type_eval
        test_result = Test_result(agentType, self.episode_ids, self.reward_history, self.asked_history,
                                  self.adviser_history, self.uncertainty,agent_a_terminal,self.agent_a.agent_type_loss,self.agent_a.agent_type_train,self.agent_a.agent_type_eval)
        print(test_result)
        return test_result
    def track_terminal(self,number_heads,agent_a_terminal,agent_b_terminal,old_v_state,action_a,action_b,i_episode):
        for i in range(number_heads):
            agent_a_terminal[i]['ep'].append(i_episode)
            agent_a_terminal[i]['mean'].append(self.agent_a.policy_net(old_v_state.data)[i][0][0][action_a].detach().numpy())
            agent_a_terminal[i]['std'].append(self.agent_a.policy_net(old_v_state.data)[i][1][0][action_a].detach().numpy())
           
        return 
    def plot_terminal_state(self,i_episode,agent_a_terminal,agent_b_terminal):
        if self.agent_a.number_heads == 1:
            i = 0
            plt.plot(agent_a_terminal[i]['ep'],agent_a_terminal[i]['mean'],'o-',label = f'agent a head {i} mean')
            plt.plot(agent_a_terminal[i]['ep'],agent_a_terminal[i]['std'],'*-',label = f'agent a head {i} std')
            plt.legend()
            plt.savefig(f'plots/mean_std_agent_a_episode_{i_episode}_{self.agent_a.agent_type_loss}{self.agent_a.agent_type_train}{self.agent_a.agent_type_eval}.png')
            plt.close()
        else:
            figa,axs_a = plt.subplots(1,self.agent_a.number_heads,figsize = (15,15))
            
            for i in range(self.agent_a.number_heads):
                axs_a[i].plot(agent_a_terminal[i]['ep'],agent_a_terminal[i]['mean'],'o-',label = f'agent a head {i} mean')
                axs_a[i].plot(agent_a_terminal[i]['ep'],agent_a_terminal[i]['std'],'o-',label = f'agent a head {i} std')
                axs_a[i].legend()
            plt.savefig(f'plots/mean_std_agent_a_episode_{i_episode}_{self.agent_a.agent_type_loss}{self.agent_a.agent_type_train}{self.agent_a.agent_type_eval}.png')
            
            plt.close(figa)
            figb,axs_b = plt.subplots(1,self.agent_b.number_heads,figsize = (15,15))
            for i in range(self.agent_b.number_heads):
                axs_b[i].plot(agent_b_terminal[i]['ep'],agent_b_terminal[i]['mean'],'o-',label = f'agent b head {i} mean')
                axs_b[i].plot(agent_b_terminal[i]['ep'],agent_b_terminal[i]['std'],'o-',label = f'agent b head {i} std')
                axs_b[i].legend()
            plt.savefig(f'plots/mean_std_agent_b_episode_{i_episode}_{self.agent_b.agent_type_loss}{self.agent_b.agent_type_train}{self.agent_b.agent_type_eval}.png')
            plt.close(figb)
        return


Test_result = namedtuple('Test_result',
                         ('AgentType', 'EPOCH_ID', 'REWARDS', 'TIMES_ASKED', 'TIMES_GIVEN', 'UNCERTAINTY','TERMINAL_TRACK_A','TERMINAL_TRACK_B','Loss','Train','Evaluation'))
Test_setup_single = namedtuple('Test_setup',
                        ('AgentType', 'NUMBER_HEADS', 'EPOCHS', 'BUFFER', 'BATCH_SIZE', 'TARGET_UPDATE', 'BUDGET', 'VA',
                         'VG','Loss','Train','Evaluation'))


def execute_test_single(test_id, test, return_dict):
    print(test)
    agenttype, number_heads, epochs, buffer, batch_size, target_update, budget, va, vg,agent_type_loss, agent_type_train,agent_type_eval = test
    print("test #: %s" % test_id)
    executor = TestExecutor(number_heads, buffer, agenttype, budget, va, vg,agent_type_loss, agent_type_train,agent_type_eval)
    return_dict[test_id] = executor.train_and_evaluate_agent(epochs, target_update, batch_size)


def mean(*vas):
    array = np.array([])
    for va in vas:
        array = np.append(array, np.asarray(va, dtype=np.float32))
    if len(array) > 0:
        return np.average(array, axis=0)
    else:
        return 0