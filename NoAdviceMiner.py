from Miner import *


class NoAdviceMiner(Miner):
    def __init__(self, number_heads, budget, va=0, vg=0):
        super(NoAdviceMiner, self).__init__(number_heads, budget, 0, 0)

    def probability_advise_in_state(self, state):
        return 0

    def probability_ask_in_state(self, env):
        return 0
