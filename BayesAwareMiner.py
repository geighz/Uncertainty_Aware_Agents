from Miner_Bayes import *
#from gridworld import v_state
# from two_goalworld import v_state
import torch.distributions as td
import numpy as np
from import_game  import *#get_grid_for_player,v_state



class BayesAwareMiner(Miner_Bayes):
    def __init__(self, number_heads, budget, va, vg):
        super(BayesAwareMiner, self).__init__(number_heads, budget, va, vg)
        
    def probability_advise_in_state(self, state):
        inverse_state = get_grid_for_player(state, np.array([0, 0, 0, 0, 1]))
        pessimistic = False
        uncertainty = self.calculate_uncertainty(v_state(inverse_state),pessimistic)
        if uncertainty < self.vg:
            return 1
        else:
            return 0
    def probability_ask_in_state(self, env):
        pessimistic=True
        uncertainty = self.calculate_uncertainty(env.v_state,pessimistic)
        if uncertainty > self.va:
            return 1
        else:
            return 0

    # This is the estimated uncertainty, uncertainty can never be calculated otherwise it wouldn't be uncertainty, but I'm not certain that this is true..
    # Decision based uncertainty 
    def calculate_uncertainty(self, v_state,pessimistic):
        qval = self.policy_net(v_state)
        actions = len(qval[0][0][0])
        disagreement_actions = torch.zeros(actions)
        min_action = np.inf
        max_action = -1
        norm_variance = 0
        
        for action in range(actions):
            min_head = np.inf
            max_head = -1
            means_action = []
            for head in qval:
                #head[0][0][action] USED FOR MEAN
                #head[1][0][action] USED FOR STD
                if head[1][0][action] >= max_head:
                    max_head = head[1][0][action]
                if head[1][0][action] <= min_head:
                    min_head = head[1][0][action]
                means_action.append(head[0][0][action])         
            norm_variance += self.variance(means_action) 
            if min_head >= max_action:
                max_action = min_head
            if max_head <= min_action:
                min_action = max_head
    
        uncertainty_measure = norm_variance+min_action if pessimistic else norm_variance + max_action
        # var_uncertainty = min_action if pessimistic else max_action
        # uncertainty = var_uncertainty+norm_variance
        self.uncertainty.append(uncertainty_measure.detach().numpy())
        return uncertainty_measure.detach().numpy()
# Unceratinty quantification using discrepancy in Wasserstein space.    
    def calculate_uncertainty_alternative(self, v_state,*arg):
        qval = self.policy_net(v_state)
        # num_heads = len(qval)
        actions = len(qval[0][0][0])
        disagreement_actions = torch.zeros(actions)
        for action in range(actions):
            #Compute the predictive normal
            # Get the mean and variance for the action at each head 
            mean_predictive = torch.zeros([1])
            var_predictive = torch.zeros([1])
            for head in qval:
                # mean,std = head[0][0][action],head[1][0][action]
                mean_predictive += head[0][0][action]
                # See page 5 Simple and Scalable predictive unceratinty...
                var_predictive += head[1][0][action]**2+head[0][0][action]**2
            # predictive_dist
            mean_predictive /= self.number_heads
            std_predictive = torch.sqrt(var_predictive/(self.number_heads)-mean_predictive**2)
            sum_w2 = torch.zeros([1])
            for head in qval:
                sum_w2 += ((head[0][0][action]-mean_predictive)**2+(head[1][0][action]-std_predictive)**2)**(0.5)
            disagreement_actions[action] =sum_w2/self.number_heads
        uncertainty =torch.mean(disagreement_actions).detach().numpy()
        self.uncertainty.append(uncertainty)
        return uncertainty

    def variance(self, predictions):
        predictions = torch.stack(predictions)
        var = predictions.var(dim=0)
        mean = abs(predictions.mean())
        return var / mean
