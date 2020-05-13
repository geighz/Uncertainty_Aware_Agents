from Miner import *


class NoAdviceMiner(Miner):
    def probability_advise_in_state(self, state):
        return 0

    def probability_ask_in_state(self, env):
        return 0
