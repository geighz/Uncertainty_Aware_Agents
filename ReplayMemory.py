import torch
import random
from collections import namedtuple
from torch.autograd import Variable

Transition = namedtuple('Transition',
                        ('state', 'action_a', 'action_b', 'new_state', 'reward', 'non_final'))


class ReplayMemory(object):

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
        actions_a = Variable(torch.LongTensor(batch.action_a)).view(-1, 1)
        actions_b = Variable(torch.LongTensor(batch.action_b)).view(-1, 1)
        new_states = Variable(torch.cat(batch.new_state))
        rewards = Variable(torch.FloatTensor(batch.reward))
        # TODO: is a long int64 really the best choice here?
        non_final = Variable(torch.LongTensor(batch.non_final))
        return states, actions_a, actions_b, new_states, rewards, non_final

    def __len__(self):
        return len(self.memory)
