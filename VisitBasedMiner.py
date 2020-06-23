from Miner import *
import math

va = 0.6
# kleiner --> weniger advice
vg = 0.25


def probability_ask_with_ypsilon(ypsilon):
    return (1 + va) ** -ypsilon


def psi_visit(number_of_visits):
    if number_of_visits <= 1:
        return 0
    return math.log(number_of_visits, 2)


def advising_probability(psi):
    return 1 - (1 + vg) ** -psi


class VisitBasedMiner(Miner):
    def __init__(self, number_heads):
        super(VisitBasedMiner, self).__init__(number_heads=number_heads)
        self.state_counter = {}

    def ypsilon_visit(self, hash_of_state):
        if hash_of_state in self.state_counter:
            number_of_visits = self.state_counter[hash_of_state]
        else:
            number_of_visits = 0
        # print("visited=%s" % number_of_visits)
        result = math.sqrt(number_of_visits)
        return result

    def probability_ask_in_state(self, env):
        # TODO: Is it necessary to convert the state to a tensor and back when hasing?
        hash_of_state = hash_state(env.state)
        ypsilon = self.ypsilon_visit(hash_of_state)
        return probability_ask_with_ypsilon(ypsilon)

    def count_state(self, state):
        hash_of_state = hash_state(state)
        if hash_of_state in self.state_counter:
            self.state_counter[hash_of_state] += 1
        else:
            self.state_counter[hash_of_state] = 1

    def probability_advise_in_state(self, state):
        inverse_state = get_grid_for_player(state, np.array([0, 0, 0, 0, 1]))
        hash_of_inverse_state = hash_state(inverse_state)
        if hash_of_inverse_state in self.state_counter:
            number_of_visits = self.state_counter[hash_of_inverse_state]
        else:
            number_of_visits = 0
        # print("visited=%s" % number_of_visits)
        psi = psi_visit(number_of_visits)
        return advising_probability(psi)

    def optimize(self, states, actions, new_states, rewards, non_final_mask):
        super(VisitBasedMiner, self).optimize(states, actions, new_states, rewards, non_final_mask)
        for state in states:
            self.count_state(state)
