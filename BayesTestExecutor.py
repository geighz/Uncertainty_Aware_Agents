from ReplayMemory import ReplayMemory
from evaluation import *
from collections import namedtuple
import numpy as np
import os
import torch
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import os.path


class TestExecutor:
    def __init__(self, number_heads, buffer, agent, budget, va, vg):
        np.random.seed()
        torch.random.seed()
        self.epsilon = 1
        self.agent_b = agent(number_heads, budget, va, vg)
        self.agent_a = agent(number_heads, budget, va, vg)
        self.agent_a.set_partner(self.agent_b)
        self.agent_b.set_partner(self.agent_a)
        self.reward_history = []
        self.episode_ids = np.array([])
        self.asked_history = np.array([])
        self.adviser_history = np.array([])
        self.memory = ReplayMemory(buffer)
        self.env = GAME_ENV()
        #self.env = TwoGoal()
        self.uncertainty = np.array([])
        self.plot = True
        # self.var 
        #self.env.render()

    def track_progress(self, episode_number):
        if episode_number % 250 == 0:
            self.episode_ids = np.append(self.episode_ids, episode_number)
            agent_a = self.agent_a
            agent_b = self.agent_b
            # if episode_number==  0 : 
            #     self.reward_history = np.append(self.reward_history,0)
            #     self.asked_history = np.append(self.asked_history, 0)
            #     self.uncertainty = np.append(self.uncertainty,0)
            #     self.adviser_history = np.append(self.adviser_history, 0)
            #     return
            mean_reward, uncertainty_mean = evaluate_agents(agent_a, agent_b)
            self.reward_history = np.append(self.reward_history, mean_reward)
            times_asked = (agent_a.times_asked + agent_b.times_asked) / 2
            self.asked_history = np.append(self.asked_history, times_asked)
            times_adviser = (agent_a.times_adviser + agent_b.times_adviser) / 2
            self.adviser_history = np.append(self.adviser_history, times_adviser)
            self.uncertainty = np.append(self.uncertainty, uncertainty_mean)
            self.vars = []

    def train_and_evaluate_agent(self, epochs, target_update, batch_size):
        terminal_heads_a_mean ={0: [],1: [], 2: [], 3:[], 4: [] }
        terminal_heads_b_mean ={0: [],1: [], 2: [], 3:[], 4: [] }
        terminal_heads_a_std ={0: [],1: [], 2: [], 3:[], 4: [] }
        terminal_heads_b_std ={0: [],1: [], 2: [], 3:[], 4: [] }
        # terminal_loss_heads_a = 40*torch.ones(epochs,5,2,dtype= torch.int)
        # terminal_loss_heads_b = 40*torch.ones(epochs,5,2,dtype = torch.int)
        # counter = 0
        done_times = 0
        for i_episode in tqdm(range(epochs),desc='Training'):
            self.track_progress(i_episode)
            tracking_time = 250
            if i_episode % tracking_time == 0 and i_episode:
                #check = self.reward_history
               
                #check = [sum(tup)/tracking_time for tup in zip(*self.agent_b.uncertainty[i_episode-tracking_time:i_episode])]
                
                print("%s Game #: %s, %f, %f,%s" % (os.getpid(), i_episode,self.reward_history[-1],self.uncertainty[-1],done_times))
                #print(check)
                # print("%s Game #: %s" % (os.getpid(), i_episode))
            self.env.reset()
            done = False
            step = 0
            loss_heads_a =torch.zeros((5,2))
            loss_heads_b =torch.zeros((5,2))
            # while game still in progress
            while not done:
                old_v_state = self.env.v_state
                action_a = self.agent_a.choose_training_action(self.env, self.epsilon)
                action_b = self.agent_b.choose_training_action(self.env, self.epsilon)
                # Take action, observe new state S'
                _, reward, done, _ = self.env.step(action_a, action_b)
                step += 1
                
                self.memory.push(old_v_state.data, action_a, action_b, self.env.v_state.data, reward, not done)
                if done:
                    done_times+=1
                # if buffer not filled, add to it
                if len(self.memory) < self.memory.capacity:
                    if done:
                        break
                    else:
                        continue
                states, actions_a, actions_b, new_states, reward, non_final = self.memory.sample(batch_size)              
                loss_heads_a = self.agent_a.optimize(states, actions_a, new_states, reward, non_final)
                loss_heads_b = self.agent_b.optimize(states, actions_b, new_states, reward, non_final)
                
                if step > 20:
                    break
            
            # for i in range(self.agent_a.number_heads):
            #     if loss_heads_a[i,0]> 0:
            #         terminal_heads_a_mean[i].append(loss_heads_a[i,0].detach().numpy())
                    
            #         terminal_heads_a_std[i].append(loss_heads_a[i,1].detach().numpy())
                    
            #     if loss_heads_b[i,0]> 0:
            #         terminal_heads_b_mean[i].append(loss_heads_b[i,0].detach().numpy())
            #         terminal_heads_b_std[i].append(loss_heads_b[i,1].detach().numpy())

            # if loss_heads_a[0,0]> 0:
            for i in range(self.agent_a.number_heads):
                terminal_heads_a_mean[i].append(loss_heads_a[i,0].detach().numpy())
                
                terminal_heads_a_std[i].append(loss_heads_a[i,1].detach().numpy())
                
                terminal_heads_b_mean[i].append(loss_heads_b[i,0].detach().numpy())
                terminal_heads_b_std[i].append(loss_heads_b[i,1].detach().numpy())

            if (i_episode%500 == 0 ):
                fig, axs = plt.subplots(2,5,figsize=(15, 15))
                for i in range(self.agent_a.number_heads):
                    
                    #terminal_loss_heads_a = 1000*torch.ones(epochs,5,2)
                    # axs[0,i].plot(20*np.ones(counter),'r-',label = 'True mean')
                    # axs[0,i].plot(0*np.ones(counter),'b-', label = 'True std')
                    axs[0,i].plot(terminal_heads_a_mean[i],label = f'agent a head {i} mean')
                    axs[0,i].plot(terminal_heads_a_std[i],label = f'agent a head {i} std')
                    
                    # axs[0,i].plot(terminal_loss_heads_a[:counter,i,1],label = f'agent a head {i} std')
                    axs[0,i].legend()
                    # axs[1,i].plot(20*np.ones(counter),'r-', label = 'True mean')
                    # axs[1,i].plot(0*np.ones(counter),'b-', label = 'True std')
                    axs[1,i].plot(terminal_heads_b_mean[i],label = f'agent b head {i} mean')
                    axs[1,i].plot(terminal_heads_b_std[i],label = f'agent b head {i} std')
                    
                    # axs[1,i].plot(terminal_loss_heads_b[:counter,i,1],label = f'agent b head {i} std')
                    axs[1,i].legend()
                # plt.show()  
                plt.savefig('plots/loss_mean_std_episode_{}.png'.format(i_episode))
                plt.close()


            if self.epsilon > 0.02:
                self.epsilon -= (1 / epochs)
            
                # self.epsilon = 1#self.epsilon
            if i_episode % target_update == 0:
                
                for head_number in range(self.agent_a.policy_net.number_heads):
                    self.agent_a.update_target_net()
                    self.agent_b.update_target_net()

        agentType = type(self.agent_a).__name__
        test_result = Test_result(agentType, self.episode_ids, self.reward_history, self.asked_history,
                                  self.adviser_history, self.uncertainty)
        print(test_result)
        return test_result


Test_result = namedtuple('Test_result',
                         ('AgentType', 'EPOCH_ID', 'REWARDS', 'TIMES_ASKED', 'TIMES_GIVEN', 'UNCERTAINTY'))
Test_setup_bayes = namedtuple('Test_setup',
                        ('AgentType', 'NUMBER_HEADS', 'EPOCHS', 'BUFFER', 'BATCH_SIZE', 'TARGET_UPDATE', 'BUDGET', 'VA',
                         'VG'))


def execute_test_bayes(test_id, test, return_dict):
    print(test)
    agenttype, number_heads, epochs, buffer, batch_size, target_update, budget, va, vg = test
    print("test #: %s" % test_id)
    executor = TestExecutor(number_heads, buffer, agenttype, budget, va, vg)
    return_dict[test_id] = executor.train_and_evaluate_agent(epochs, target_update, batch_size)


def mean(*vas):
    array = np.array([])
    for va in vas:
        array = np.append(array, np.asarray(va, dtype=np.float32))
    if len(array) > 0:
        return np.average(array, axis=0)
    else:
        return 0