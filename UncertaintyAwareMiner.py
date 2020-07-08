from Miner import *
from gridworld import v_state


def variance(predictions):
    predictions = torch.stack(predictions)
    var = predictions.var(dim=0)
    return var


class UncertaintyAwareMiner(Miner):
    def __init__(self, number_heads, budget, va, vg):
        super(UncertaintyAwareMiner, self).__init__(number_heads, budget, va, vg)
        self.va_history = []
        self.vg_history = []

    def probability_advise_in_state(self, state):
        # TODO: should get_grid_for_player be a function call higher?
        inverse_state = get_grid_for_player(state, np.array([0, 0, 0, 0, 1]))
        uncertainty = self.calculate_uncertainty(v_state(inverse_state))
        self.vg_history.append(uncertainty)
        if uncertainty < self.vg:
            return 1
        else:
            return 0

    def probability_ask_in_state(self, env):
        uncertainty = self.calculate_uncertainty(env.v_state)
        self.va_history.append(uncertainty)
        if uncertainty > self.va:
            return 1
        else:
            return 0

    # This is the estimated uncertainty, uncertainty can never be calculated otherwise it wouldn't be uncertainty
    def calculate_uncertainty(self, v_state):
        qval = self.policy_net(v_state)
        sum_variance = 0
        for action in range(4):
            # TODO: Use gather instead of iterating
            # TODO: Find the right name for the fat printed Q from page 5 of the "Uncertainty-Aware..." paper
            predictions = []
            for i in range(self.number_heads):
                state_action_value = qval[i][0][action].data
                predictions.append(state_action_value)
            sum_variance += variance(predictions)
        return sum_variance / 4
