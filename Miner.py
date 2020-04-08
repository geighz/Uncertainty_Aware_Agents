import math
from gridworld import *
from torch.autograd import Variable
import torch

va = 0.6
vg = 0.25
state_counter = {}


def probability_ask_with_ypsilon(ypsilon):
    return (1 + va) ** -ypsilon


def advising_probability(psi):
    return 1 - (1 + vg) ** -psi


def give_advise(state, model):
    inv_state = get_grid_for_player(state, np.array([0, 0, 0, 0, 1]))
    v_state = Variable(torch.from_numpy(inv_state)).view(1, -1)
    q_values = model(v_state)
    action = np.argmax(q_values.data)
    return action

def probability_ask_with_state(state):
    # TODO: Is it necessary to convert the state to a tensor and back when hasing?
    v_state = Variable(torch.from_numpy(state)).view(1, -1)
    hash_of_state = hash_state(v_state)
    ypsilon = ypsilon_visit(hash_of_state)
    return probability_ask_with_ypsilon(ypsilon)

advising_dic = {}


def advising_probability_in_state(state):
    inverse_state = get_grid_for_player(state, np.array([0, 0, 0, 0, 1]))
    v_inverse_state = Variable(torch.from_numpy(inverse_state)).view(1, -1)
    hash_of_inverse_state = hash_state(v_inverse_state)
    if hash_of_inverse_state in state_counter:
        number_of_visits = state_counter[hash_of_inverse_state]
    else:
        number_of_visits = 0
    # print("visited=%s" % number_of_visits)
    psi = psi_visit(number_of_visits)
    return advising_probability(psi)


def hash_state(state):
    hash = list(state[0].numpy().astype(int))
    hash = bin(int(''.join(map(str, hash)), 2) << 1)
    return hash


def ypsilon_visit(hash_of_state):
    if hash_of_state in state_counter:
        number_of_visits = state_counter[hash_of_state]
    else:
        number_of_visits = 0
    # print("visited=%s" % number_of_visits)
    result = math.sqrt(number_of_visits)
    return result


def psi_visit(number_of_visits):
    if number_of_visits <= 1:
        # TODO fix this workaround, normally it should be minus infinity
        return 0
    return math.log(number_of_visits, 2)


def count_state(state):
    v_state = Variable(torch.from_numpy(state)).view(1, -1)
    hash_of_state = hash_state(v_state)
    if hash_of_state in state_counter:
        state_counter[hash_of_state] += 1
    else:
        state_counter[hash_of_state] = 1


def exploration_strategy(qval, epsilon=1):
    # choose random action
    if np.random.random() < epsilon:
        action = np.random.randint(0, 4)
        # print("A takes random action {}".format(action_a))
    else:  # choose best action from Q(s,a) values
        action = np.argmax(qval.data)
        # print("A takes best action {}".format(action_a))
    return action
