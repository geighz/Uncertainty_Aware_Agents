from ReplayMemory import ReplayMemory
from evaluation_gold import *
from collections import namedtuple
import numpy as np
import os
import torch
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import os.path


class TestExecutor:
    def __init__(self, number_heads, buffer, agent, budget, va, vg,agent_type_loss, agent_type_train,agent_type_eval):
        np.random.seed()
        torch.random.seed()
        self.epsilon = 1
        self.agent_b = agent(number_heads, budget, va, vg,agent_type_loss, agent_type_train,agent_type_eval)
        self.agent_a = agent(number_heads, budget, va, vg,agent_type_loss, agent_type_train,agent_type_eval)
        self.agent_a.set_partner(self.agent_b)
        self.agent_b.set_partner(self.agent_a)
        self.reward_history = []
        self.episode_ids = np.array([])
        self.asked_history = np.array([])
        self.adviser_history = np.array([])
        self.memory = ReplayMemory(buffer)
        self.env = GAME_ENV()
        self.uncertainty = np.array([])
        self.plot = True
        # Render env
        #self.env.render()

    def track_progress(self, episode_number):
        if episode_number % 100 == 0 and episode_number >0:
            self.episode_ids = np.append(self.episode_ids, episode_number)
            agent_a = self.agent_a
            agent_b = self.agent_b
            mean_reward, uncertainty_mean = evaluate_agents(agent_a, agent_b)
            self.reward_history = np.append(self.reward_history, mean_reward)
            times_asked = (agent_a.times_asked + agent_b.times_asked) / 2
            self.asked_history = np.append(self.asked_history, times_asked)
            times_adviser = (agent_a.times_adviser + agent_b.times_adviser) / 2
            self.adviser_history = np.append(self.adviser_history, times_adviser)
            self.uncertainty = np.append(self.uncertainty, uncertainty_mean)
            # self.vars = []

    def train_and_evaluate_agent(self, epochs, target_update, batch_size):
        agent_a_terminal = {}
        agent_b_terminal = {}
        for i in range(self.agent_a.number_heads):
            agent_a_terminal[i] = {'mean':[],'std':[]}
            agent_b_terminal[i] = {'mean':[],'std':[]}
        
        
        done_times = 0
        for i_episode in tqdm(range(epochs),desc='Training'):
            # There's a problem here for sure
            self.track_progress(i_episode)

            tracking_time = 100
            if i_episode % tracking_time == 0 and i_episode >0:
                
                # continue
                print("%s Game #: %s, %f, %f,%s" % (os.getpid(), i_episode,self.reward_history[-1],self.uncertainty[-1],done_times))
                
            self.env.reset()
            done = False
            step = 0
            # means_std_a_ep =torch.zeros((5,2))
            # means_std_b_ep =torch.zeros((5,2))
            # while game still in progress
            while not done:
                old_v_state = self.env.v_state
                # t1 = time.time()
                action_a = self.agent_a.choose_training_action(self.env, self.epsilon)
                action_b = self.agent_b.choose_training_action(self.env, self.epsilon)
                # Take action, observe new state S'
                _, reward, done, _ = self.env.step(action_a, action_b)
                step += 1
                # print(f'time training action computation {t1 - time.time()}')
                self.memory.push(old_v_state.data, action_a, action_b, self.env.v_state.data, reward, not done)
                #if done and reward > 0:
                    #TODO
                    #track_terminal()
                
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
                self.agent_b.optimize(states, actions_b, new_states, reward, non_final)
                # print(f'time optimize computation {t1 - time.time()}')
                if step > 20:
                    break
            

            # if (i_episode%2000 == 0 ):
                #TODO
                # plot_terminal_state()
            


            if self.epsilon > 0.02:
                self.epsilon -= (1 / epochs)
            
            if i_episode % target_update == 0:
                
                for head_number in range(self.agent_a.policy_net.number_heads):
                    self.agent_a.update_target_net()
                    self.agent_b.update_target_net()

        agentType = 'PNN-DQN'+self.agent_a.agent_type_loss+self.agent_a.agent_type_train+self.agent_a.agent_type_eval
        test_result = Test_result(agentType, self.episode_ids, self.reward_history, self.asked_history,
                                  self.adviser_history, self.uncertainty)
        print(test_result)
        return test_result
    def track_termianl():
        #TODO
        #    qval_end_a = self.agent_a.policy_net(old_v_state.data)[action_a]
        #    qval_end_b = self.agent_b.policy_net(old_v_state.data)[action_b]
        #    for i in range(self.agent_a.number_heads):
        #        means_a[i].append(qval_end_a[i,0].detach().numpy())
        #        std_a[i].append(qval_end_a[i,1].detach().numpy())
        #        means_b[i].append(qval_end_b[i,0].detach().numpy())
        #        std_b[i].append(qval_end_b[i,1].detach().numpy())
        return 
    
    def plot_terminal_state():
        #TODO
        #     figa, axs_a = plt.subplots(1,self.agent_a.number_heads,figsize=(15, 15))
        #     figb, axs_b = plt.subplots(1,self.agent_a.number_heads,figsize=(15, 15))
        #     if self.agent_a.number_heads == 1:
        #         i = 0
        #         axs_a.plot(means_a[i],label = f'agent a head {i} mean')
        #         axs_a.plot(std_a[i],label = f'agent a head {i} std')
        #         axs_a.legend()
        #         plt.savefig('plots/mean_std_agent_a_episode_{}_{}{}{}.png'.format(i_episode,self.agent_a.agent_type_loss,self.agent_a.agent_type_train,self.agent_a.agent_type_eval))
        #         plt.close()
        #         axs_b.plot(means_b[i],label = f'agent b head {i} mean')
        #         axs_b.plot(std_b[i],label = f'agent b head {i} std')
        #         axs_b.legend()

            
        #         plt.savefig('plots/mean_std_agent_b_episode_{}_{}{}{}.png'.format(i_episode,self.agent_a.agent_type_loss,self.agent_a.agent_type_train,self.agent_a.agent_type_eval))
        #         plt.close()
        #     else :
        #         for i in range(self.agent_a.number_heads):
                
        #             axs_a[i].plot(means_a[i],label = f'agent a head {i} mean')
        #             axs_a[i].plot(std_a[i],label = f'agent a head {i} std')
        #             axs_a[i].legend()
        #         plt.savefig('plots/mean_std_agent_a_episode_{}_{}{}{}.png'.format(i_episode,self.agent_a.agent_type_loss,self.agent_a.agent_type_train,self.agent_a.agent_type_eval))
        #         plt.close()

        #         for i in range(self.agent_a.number_heads):
        #             axs_b[i].plot(means_b[i],label = f'agent b head {i} mean')
        #             axs_b[i].plot(std_b[i],label = f'agent b head {i} std')
        #             axs_b[i].legend()
                
        #         plt.savefig('plots/mean_std_agent_b_episode_{}_{}{}{}.png'.format(i_episode,self.agent_a.agent_type_loss,self.agent_a.agent_type_train,self.agent_a.agent_type_eval))
        #         plt.close()
        return


Test_result = namedtuple('Test_result',
                         ('AgentType', 'EPOCH_ID', 'REWARDS', 'TIMES_ASKED', 'TIMES_GIVEN', 'UNCERTAINTY'))
Test_setup_bayes = namedtuple('Test_setup',
                        ('AgentType', 'NUMBER_HEADS', 'EPOCHS', 'BUFFER', 'BATCH_SIZE', 'TARGET_UPDATE', 'BUDGET', 'VA',
                         'VG','Loss','Train','Evaluation'))


def execute_test_bayes(test_id, test, return_dict):
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