from DQN import ReplayMemory, Transition, hidden_unit, Q_learning
from torch.autograd import Variable
from gridworld import *
import torch.optim as optim
import torch

## Include the replay experience

epochs = 10
gamma = 0.9  # since it may take several moves to goal, making gamma high
epsilon = 1
model = Q_learning(80, [164, 150], 16, hidden_unit)
# for hidden in model.hidden_units:
#     print(hidden.nn.weight.size())
#     print(hidden.nn.weight)
# print(model.final_unit.weight.size())
# print(model.final_unit.weight)
optimizer = optim.RMSprop(model.parameters(), lr=0.001)
# optimizer = optim.SGD(model.parameters(), lr=0.1)
loss = torch.nn.MSELoss()
# buffer = 80
# BATCH_SIZE = 10
# memory = ReplayMemory(buffer)

for i in range(epochs):
    print("Game #: %s" % (i,))
    state = init_grid_player()
    print(disp_grid(state))
    status = 1
    step = 0
    # while game still in progress
    while status == 1:
        v_state = Variable(torch.from_numpy(state)).view(1, -1)
        qval = model(v_state)
        print(qval)
        if np.random.random() < epsilon:  # choose random action
            action = np.random.randint(0, 16)
            print("Take random action {}".format(action))
        else:  # choose best action from Q(s,a) values
            action = np.argmax(qval.data)
            print("Take best action {}".format(action))
        # Take action, observe new state S'
        action_a = action // 4
        action_b = action % 4
        new_state = make_move(state, action_a, action_b)
        # Observe reward
        reward = get_reward(new_state)
        print("reward: {}".format(reward))
        print("New state:\n", disp_grid(new_state))
        print("\n")
        step += 1
        v_new_state = Variable(torch.from_numpy(new_state)).view(1, -1)
        newQ = model(v_new_state)
        maxQ = newQ.max(1)[0].detach()
        target = qval.clone().detach()
        if reward == -1:  # non-terminal state
            update = (reward + (gamma * maxQ))
        else:  # terminal state
            update = reward
        target[0][action] = update  # target output
        print("Adjust\n{}\ntowards\n{}".format(qval, target))
        # dies sorgt für einen backward pass nicht nur durch qval sondern auch durch target
        output = loss(qval, target)

        # Optimize the model
        # Clear gradients of all optimized torch.Tensor s.
        optimizer.zero_grad()
        # compute gradients
        output.backward()
        # Gradient clipping can keep things stable.
        # for p in model.parameters():
        #     p.grad.data.clamp_(-1, 1)
        # update model parameters
        optimizer.step()
        newqval = model(v_state)
        print("New Qval\n{}".format(newqval))
        difference = qval - newqval
        print(difference)
        print()

        state = new_state
        if reward != -1:
            status = 0
        if step > 20:
            break
    if epsilon > 0.1:
        epsilon -= (1 / epochs)


# Here is the test of AI
def testAlgo(init=0):
    i = 0
    if init == 0:
        state = initGrid()
    elif init == 1:
        state = init_grid_player()
    elif init == 2:
        state = init_grid_rand()

    print("Initial State:")
    print(disp_grid(state))
    status = 1
    # while game still in progress
    while (status == 1):
        v_state = Variable(torch.from_numpy(state))
        qval = model(v_state.view(80))
        print(qval)
        action = np.argmax(qval.data)  # take action with highest Q-value
        print('Move #: %s; Taking action: %s' % (i, action))
        action_a = action // 4
        action_b = action % 4
        state = make_move(state, action_a, action_b)
        print(disp_grid(state))
        reward = get_reward(state)
        if reward != -1:
            status = 0
            print("Reward: %s" % (reward,))
        i += 1  # If we're taking more than 10 actions, just stop, we probably can't win this game
        if i > 10:
            print("Game lost; too many moves.")
            break


for i in range(20):
    testAlgo(init=1)
