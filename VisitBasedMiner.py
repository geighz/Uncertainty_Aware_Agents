from Miner import *
import math

# higher va --> lower asking for advice
va = 0.6
# "while a higher vg results in a higher probability of giving advice"
# lower vg --> give fewer advice
vg = 0.25


def asking_probability(ypsilon):
    return (1 + va) ** -ypsilon


def advising_probability(psi):
    return 1 - (1 + vg) ** -psi


class VisitBasedMiner(Miner):
    def __init__(self, number_heads, budget):
        super(VisitBasedMiner, self).__init__(number_heads, budget)
        self.state_counter = {}

    def count_state(self, state):
        hash_of_state = hash_state(state)
        if hash_of_state in self.state_counter:
            self.state_counter[hash_of_state] += 1
        else:
            self.state_counter[hash_of_state] = 1

    def times_visited(self, state):
        hash_of_state = hash_state(state)
        if hash_of_state in self.state_counter:
            return self.state_counter[hash_of_state]
        else:
            return 0

    def ypsilon(self, state):
        n = self.times_visited(state)
        return math.sqrt(n)

    def probability_ask_in_state(self, env):
        ypsilon = self.ypsilon(env.state)
        return asking_probability(ypsilon)

    def psi(self, state):
        number_of_visits = self.times_visited(state)
        if number_of_visits <= 1:
            return 0
        return math.log(number_of_visits, 2)

    def probability_advise_in_state(self, state):
        inverse_state = get_grid_for_player(state, np.array([0, 0, 0, 0, 1]))
        psi = self.psi(inverse_state)
        return advising_probability(psi)

    def optimize(self, states, actions, new_states, rewards, non_final_mask):
        super(VisitBasedMiner, self).optimize(states, actions, new_states, rewards, non_final_mask)
        for state in states:
            self.count_state(state)
