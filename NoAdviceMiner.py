from Miner import *


class NoAdviceMiner(Miner):
    def __init__(self, number_heads):
        super(NoAdviceMiner, self).__init__(number_heads=number_heads)

    def probability_advise_in_state(self, state):
        return 0

    def probability_ask_in_state(self, env):
        return 0
