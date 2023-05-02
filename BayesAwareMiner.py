from Miner_Bayes import *
from gridworld import v_state
import torch.distributions as td



class BayesAwareMiner(Miner_Bayes):
    def __init__(self, number_heads, budget, va, vg):
        super(BayesAwareMiner, self).__init__(number_heads, budget, va, vg)
    #GOOD
    def probability_advise_in_state(self, state):
        inverse_state = get_grid_for_player(state, np.array([0, 0, 0, 0, 1]))
        uncertainty = self.calculate_uncertainty(v_state(inverse_state))
        if uncertainty < self.vg:
            return 1
        else:
            return 0
    #LIKELY TO BE GOOD
    def probability_ask_in_state(self, env):
        uncertainty = self.calculate_uncertainty(env.v_state)
        if uncertainty > self.va:
            return 1
        else:
            return 0

    # This is the estimated uncertainty, uncertainty can never be calculated otherwise it wouldn't be uncertainty
    #NEEDS TO CHANGE
    def calculate_uncertainty(self, v_state):
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
            check = var_predictive/(self.number_heads)-mean_predictive**2
            std_predictive = torch.sqrt(var_predictive/(self.number_heads)-mean_predictive**2)
            predict_dist = td.Normal(mean_predictive,std_predictive)
            sum_kl = torch.zeros([1])
            for head in qval:
                # check=qval[head][0][0][action]
                normal_head = td.Normal(head[0][0][action],head[1][0][action])
                sum_kl +=  td.kl.kl_divergence(normal_head, predict_dist)
            disagreement_actions[action] =sum_kl

        uncertainty = torch.mean(disagreement_actions).detach().numpy()
        self.uncertainty.append(uncertainty)
        return uncertainty
    #NEEDS TO CHANGE
    # def variance(self, predictions):
    #     predictions = torch.stack(predictions)
    #     var = predictions.var(dim=0)
    #     return var
