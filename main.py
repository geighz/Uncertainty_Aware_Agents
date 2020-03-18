from DQN import ReplayMemory, Transition, hidden_unit, Q_learning
from torch.autograd import Variable
from gridworld import *
import torch.optim as optim
import torch

## Include the replay experience

epochs = 1000
gamma = 0.9  # since it may take several moves to goal, making gamma high
epsilon = 1
model = Q_learning(64, [164, 150], 4, hidden_unit)
optimizer = optim.RMSprop(model.parameters(), lr=1e-3)
# optimizer = optim.SGD(model.parameters(), lr = 0.1, momentum = 0)
criterion = torch.nn.MSELoss()
buffer = 80
BATCH_SIZE = 10
memory = ReplayMemory(buffer)

for i in range(epochs):
    print("Game #: %s" % (i,))
    state = initGridPlayer()
    print(dispGrid(state))
    status = 1
    step = 0
    # while game still in progress
    while (status == 1):
        v_state = Variable(torch.from_numpy(state)).view(1, -1)
        qval = model(v_state)
        print(qval)
        if (np.random.random() < epsilon):  # choose random action
            action = np.random.randint(0, 4)
            print("Take random action {}".format(action))
        else:  # choose best action from Q(s,a) values
            action = np.argmax(qval.data)
            print("Take best action {}".format(action))
        # Take action, observe new state S'
        new_state = makeMove(state, action)
        # Observe reward
        reward = getReward(new_state)
        print("reward: {}".format(reward))
        print("New state:\n", dispGrid(new_state))
        print("\n")
        step += 1
        v_new_state = Variable(torch.from_numpy(new_state)).view(1, -1)
        newQ = model(v_new_state)
        maxQ = newQ.max(1)[0]
        target = qval.clone()
        if reward == -1:  # non-terminal state
            update = (reward + (gamma * maxQ))
        else:  # terminal state
            update = reward
        target[0][action] = update  # target output

        loss = criterion(qval, target)

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        for p in model.parameters():
            p.grad.data.clamp_(-1, 1)
        optimizer.step()

        state = new_state
        if reward != -1:
            status = 0
        if step > 20:
            break
    if epsilon > 0.1:
        epsilon -= (1 / epochs)


## Here is the test of AI
def testAlgo(init=0):
    i = 0
    if init == 0:
        state = initGrid()
    elif init == 1:
        state = initGridPlayer()
    elif init == 2:
        state = initGridRand()

    print("Initial State:")
    print(dispGrid(state))
    status = 1
    # while game still in progress
    while (status == 1):
        v_state = Variable(torch.from_numpy(state))
        qval = model(v_state.view(64))
        print(qval)
        action = np.argmax(qval.data)  # take action with highest Q-value
        print('Move #: %s; Taking action: %s' % (i, action))
        state = makeMove(state, action)
        print(dispGrid(state))
        reward = getReward(state)
        if reward != -1:
            status = 0
            print("Reward: %s" % (reward,))
        i += 1  # If we're taking more than 10 actions, just stop, we probably can't win this game
        if (i > 10):
            print("Game lost; too many moves.")
            break


testAlgo(init=0)
