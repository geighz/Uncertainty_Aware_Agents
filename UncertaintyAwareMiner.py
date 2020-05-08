from Miner import *
from gridworld import v_state

UncertaintyThreshold = 0.11


def normalized_variance(predictions):
    predictions = torch.stack(predictions)
    var = predictions.var(dim=0)
    mean = predictions.mean()
    return var/mean


class UncertaintyAwareMiner(Miner):
    def probability_advise_in_state(self, state):
        # TODO: should get_grid_for_player be a function call higher?
        inverse_state = get_grid_for_player(state, np.array([0, 0, 0, 0, 1]))
        uncertainty = self.calculate_uncertainty(v_state(inverse_state))
        if uncertainty < UncertaintyThreshold:
            return 1
        else:
            return 0

    def probability_ask_in_state(self, env):
        uncertainty = self.calculate_uncertainty(env.v_state)
        if uncertainty > UncertaintyThreshold:
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
            sum_variance += normalized_variance(predictions)
        return sum_variance / 4
