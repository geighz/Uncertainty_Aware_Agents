from Miner import *
import math


class VisitBasedMiner(Miner):
    # higher va --> lower asking for advice
    # "a higher vg results in a higher probability of giving advice"
    # lower vg --> give fewer advice

    def __init__(self, number_heads, budget, va, vg):
        super(VisitBasedMiner, self).__init__(number_heads, budget, va, vg)
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

    def asking_probability(self, ypsilon):
        return (1 + self.va) ** -ypsilon

    def ypsilon(self, state):
        n = self.times_visited(state)
        return math.sqrt(n)

    def probability_ask_in_state(self, env):
        ypsilon = self.ypsilon(env.state)
        return self.asking_probability(ypsilon)

    def advising_probability(self, psi):
        return 1 - (1 + self.vg) ** -psi

    def psi(self, state):
        number_of_visits = self.times_visited(state)
        if number_of_visits <= 1:
            return 0
        return math.log(number_of_visits, 2)

    def probability_advise_in_state(self, state):
        inverse_state = get_grid_for_player(state, np.array([0, 0, 0, 0, 1]))
        psi = self.psi(inverse_state)
        return self.advising_probability(psi)

    def optimize(self, states, actions, new_states, rewards, non_final_mask):
        super(VisitBasedMiner, self).optimize(states, actions, new_states, rewards, non_final_mask)
        for state in states:
            self.count_state(state)
