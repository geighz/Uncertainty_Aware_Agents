from VisitBasedMiner import *


class TDMiner(VisitBasedMiner):
    va = 0.6
    vg = 0.25

    def psi(self, state):
        ypsilon = self.ypsilon(state)
        state = v_state(state)
        maxQ = self.target_net.q_circumflex(state).max(1)[0]
        minQ = self.target_net.q_circumflex(state).min(1)[0]
        return ypsilon * abs(maxQ - minQ)
