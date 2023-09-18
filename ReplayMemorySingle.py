import torch
import random
from collections import namedtuple
from torch.autograd import Variable

Transition = namedtuple('Transition',
                        ('state', 'action', 'new_state', 'reward', 'non_final'))


class ReplayMemorySingle(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample_transitions(self, batch_size):
        return random.sample(self.memory, batch_size)

    def sample(self, batch_size):
        transitions = self.sample_transitions(batch_size)
        batch = Transition(*zip(*transitions))
        states = Variable(torch.cat(batch.state))
        actions = Variable(torch.LongTensor(batch.action)).view(-1, 1)
        new_states = Variable(torch.cat(batch.new_state))
        rewards = Variable(torch.FloatTensor(batch.reward))
        non_final = Variable(torch.BoolTensor(batch.non_final))
        return states, actions, new_states, rewards, non_final

    def __len__(self):
        return len(self.memory)
