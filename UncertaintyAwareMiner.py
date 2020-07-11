from Miner import *
from gridworld import v_state


class UncertaintyAwareMiner(Miner):
    def __init__(self, number_heads, budget, va, vg):
        super(UncertaintyAwareMiner, self).__init__(number_heads, budget, va, vg)
        self.uncertainty_ask = []
        self.uncertainty_give = []

    def probability_advise_in_state(self, state):
        inverse_state = get_grid_for_player(state, np.array([0, 0, 0, 0, 1]))
        uncertainty = self.calculate_uncertainty(v_state(inverse_state))
        self.uncertainty_give.append(uncertainty)
        if uncertainty < self.vg:
            return 1
        else:
            return 0

    def probability_ask_in_state(self, env):
        uncertainty = self.calculate_uncertainty(env.v_state)
        self.uncertainty_ask.append(uncertainty)
        if uncertainty > self.va:
            return 1
        else:
            return 0

    # This is the estimated uncertainty, uncertainty can never be calculated otherwise it wouldn't be uncertainty
    def calculate_uncertainty(self, v_state):
        qval_per_head = self.policy_net(v_state)
        sum_variance = 0
        for action in range(4):
            # TODO: Find the right name for the fat printed Q from page 5 of the "Uncertainty-Aware..." paper
            predictions = [qval[0][action].data for qval in qval_per_head]
            sum_variance += self.variance(predictions)
        return sum_variance / 4

    def variance(self, predictions):
        predictions = torch.stack(predictions)
        var = predictions.var(dim=0)
        return var
