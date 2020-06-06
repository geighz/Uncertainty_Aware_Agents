from UncertaintyAwareMiner import *

UncertaintyThreshold = 0.11


def normalized_variance(predictions):
    predictions = torch.stack(predictions)
    var = predictions.var(dim=0)
    mean = predictions.mean()
    return var/mean


class UncertaintyAwareMinerNormalised(UncertaintyAwareMiner):

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
