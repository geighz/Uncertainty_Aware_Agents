import torch
import torch.nn as nn
import torch.nn.functional as F


class hidden_unit(nn.Module):
    def __init__(self, in_channels, out_channels, activation):
        super(hidden_unit, self).__init__()
        self.activation = activation
        # linear transformation to the incoming data
        self.nn = nn.Linear(in_channels, out_channels)
        nn.init.normal_(self.nn.weight, std=0.15)

    def forward(self, x):
        out = self.nn(x)
        out = self.activation(out)
        return out


class Body_net(nn.Module):
    def __init__(self, in_channels, hidden_layers, out_channels, unit=hidden_unit, activation=F.relu):
        super(Body_net, self).__init__()
        assert type(hidden_layers) is list
        self.hidden_units = nn.ModuleList()
        self.in_channels = in_channels
        self.activation = activation
        hidden_layers = hidden_layers + [out_channels]
        for out_channels in hidden_layers:
            self.hidden_units.append(unit(in_channels, out_channels, activation))
            in_channels = out_channels

    def forward(self, x):
        out = x.view(-1, self.in_channels).float()
        for unit in self.hidden_units:
            out = unit(out)
        return out


class Head_net(nn.Module):
    def __init__(self, net, final_layers, out_channels, unit=hidden_unit, activation=F.relu):
        super(Head_net, self).__init__()
        self.net = net
        self.final_units = nn.ModuleList()
        last_index = len(self.net.hidden_units) - 1
        self.in_channels = self.net.hidden_units[last_index].nn.out_features
        prev_layer = self.in_channels
        for layer in final_layers:
            self.final_units.append(unit(prev_layer, layer, activation))
            prev_layer = layer
        self.final_unit = nn.Linear(prev_layer, out_channels)
        nn.init.normal_(self.final_unit.weight, std=0.15)

    def forward(self, x):
        out = self.net(x)
        for unit in self.final_units:
            out = unit(out)
        out = self.final_unit(out)
        return out


class Bootstrapped_DQN(nn.Module):
    def __init__(self, number_heads, in_channels, hidden_layers, out_channels, unit=hidden_unit, activation=F.relu):
        super(Bootstrapped_DQN, self).__init__()
        self.number_heads = number_heads
        out_body = hidden_layers.pop()
        # TODO: make NN architecture more flexible to inits
        # bc I pop a layer from the hidden layers into the head, I have more diversity across heads
        head_hidden_layers = [hidden_layers.pop()]
        body = Body_net(in_channels, hidden_layers, out_body, unit, activation)
        self.nets = []
        for i in range(self.number_heads):
            self.nets.append(Head_net(body, head_hidden_layers, out_channels))

    def load_weights(self, filename):
        for i in range(len(self.nets)):
            self.nets[i].load_state_dict(torch.load(f"{filename}.{i}"))
            self.nets[i].eval()

    def save_weights(self, filename):
        for i in range(len(self.nets)):
            torch.save(self.nets[i].state_dict(), f"{filename}.{i}")

    def print_weights(self):
        for net in self.nets:
            print(list(net.parameters()))

    def forward(self, x):
        result = []
        for i in range(self.number_heads):
            result.append(self.nets[i](x))
        return result

    def q_circumflex(self, x):
        qval = self.__call__(x)
        # q-values derived from all heads, compare with
        # Uncertainty-Aware Action Advising for Deep Reinforcement Learning Agents
        # page 5
        # mean = torch.mean(torch.stack(qval))
        sum = qval[0]
        for i in range(self.number_heads - 1):
            sum += qval[i + 1]
        return sum / self.number_heads
