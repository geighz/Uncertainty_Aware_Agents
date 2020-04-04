from DQN import ReplayMemory, Transition, hidden_unit, Q_learning
from torch.autograd import Variable
from gridworld import *
import torch.optim as optim
import torch
import matplotlib.pyplot as plt

episode_durations = []


def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    plt.pause(0.1)  # pause a bit so that plots are updated

# Here is the test of AI
def testAlgo(init=0):
    if init == 0:
        state = initGrid()
    elif init == 1:
        state = init_grid_player()
    elif init == 2:
        state = init_grid_rand()

    # print("Initial State:")
    # print(disp_grid(state))
    i = 0
    status = 1
    reward_sum = 0
    # while game still in progress
    while (status == 1):
        v_state = Variable(torch.from_numpy(state))
        qval_a = model_a(v_state.view(80))
        qval_b = model_b(v_state.view(80))
        # print(qval_a)
        # print(qval_b)
        action_a = np.argmax(qval_a.data)  # take action with highest Q-value
        action_b = np.argmax(qval_b.data)  # take action with highest Q-value
        # print('A: Move #: %s; Taking action: %s' % (i, action_a))
        # print('B: Move #: %s; Taking action: %s' % (i, action_b))
        state = make_move(state, action_a, action_b)
        # print(disp_grid(state))
        reward = get_reward(state)
        reward_sum += reward
        if reward != -2:
            status = 0
            return reward_sum
            # print("Reward: %s" % (reward,))
        i += 1  # If we're taking more than 10 actions, just stop, we probably can't win this game
        if i > 10:
            # print("Reward: Game lost; too many moves.")
            return reward_sum


epochs = 1005
gamma = 0.9  # since it may take several moves to goal, making gamma high
epsilon = 1
model_a = Q_learning(80, [164, 150], 4, hidden_unit)
model_b = Q_learning(80, [164, 150], 4, hidden_unit)
# for hidden in model.hidden_units:
#     print(hidden.nn.weight.size())
#     print(hidden.nn.weight)
# print(model.final_unit.weight.size())
# print(model.final_unit.weight)
# optimizer_a = optim.RMSprop(model_a.parameters(), lr=0.001)
# optimizer_b = optim.RMSprop(model_b.parameters(), lr=0.001)
optimizer_a = optim.Adam(model_a.parameters(), lr=0.001)
optimizer_b = optim.Adam(model_b.parameters(), lr=0.001)
# optimizer_a = optim.SGD(model_a.parameters(), lr=0.02)
# optimizer_b = optim.SGD(model_b.parameters(), lr=0.02)
loss_a = torch.nn.MSELoss()
loss_b = torch.nn.MSELoss()
# buffer = 80
# BATCH_SIZE = 10
# memory = ReplayMemory(buffer)

for i in range(epochs):
    # print("Game #: %s" % (i,))
    state = init_grid_player()
    # print(disp_grid(state))
    game_over = False
    step = 0
    # while game still in progress
    while not game_over:
        v_state = Variable(torch.from_numpy(state)).view(1, -1)
        qval_a = model_a(v_state)
        qval_b = model_b(v_state)
        # print(qval_a)
        # print(qval_b)
        if np.random.random() < epsilon:  # choose random action
            action_a = np.random.randint(0, 4)
            # print("A takes random action {}".format(action_a))
        else:  # choose best action from Q(s,a) values
            action_a = np.argmax(qval_a.data)
            # print("A takes best action {}".format(action_a))
        if np.random.random() < epsilon:  # choose random action
            action_b = np.random.randint(0, 4)
            # print("B takes random action {}".format(action_b))
        else:  # choose best action from Q(s,a) values
            action_b = np.argmax(qval_b.data)
            # print("B takes best action {}".format(action_b))
        # Take action, observe new state S'
        new_state = make_move(state, action_a, action_b)
        # Observe reward
        reward = get_reward(new_state)
        # print("reward: {}".format(reward))
        # print("New state:\n", disp_grid(new_state))
        # print("\n")
        step += 1
        v_new_state = Variable(torch.from_numpy(new_state)).view(1, -1)
        newQ_a = model_a(v_new_state)
        maxQ_a = newQ_a.max(1)[0].detach()
        newQ_b = model_b(v_new_state)
        maxQ_b = newQ_b.max(1)[0].detach()
        target_a = qval_a.clone().detach()
        target_b = qval_b.clone().detach()
        if reward == -2:  # non-terminal state
            update_a = (reward + (gamma * maxQ_a))
            update_b = (reward + (gamma * maxQ_b))
        else:  # terminal state
            update_a = reward
            update_b = reward
        target_a[0][action_a] = update_a  # target output
        target_b[0][action_b] = update_b  # target output
        # print("A: Adjust\n{}\ntowards\n{}".format(qval_a, target_a))
        # print("B: Adjust\n{}\ntowards\n{}".format(qval_b, target_b))
        # dies sorgt für einen backward pass nicht nur durch qval sondern auch durch target
        output_a = loss_a(qval_a, target_a)
        output_b = loss_b(qval_b, target_b)

        # Optimize the model
        # Clear gradients of all optimized torch.Tensor s.
        optimizer_a.zero_grad()
        optimizer_b.zero_grad()
        # compute gradients
        output_a.backward()
        output_b.backward()
        # Gradient clipping can keep things stable.
        # for p in model.parameters():
        #     p.grad.data.clamp_(-1, 1)
        # update model parameters
        optimizer_a.step()
        optimizer_b.step()
        # newqval_a = model_a(v_state)
        # newqval_b = model_b(v_state)
        # print("New Qval\n{}".format(newqval_a))
        # print("New Qval\n{}".format(newqval_b))
        # difference_a = qval_a - newqval_a
        # difference_b = qval_b - newqval_b
        # print(difference_a)
        # print(difference_b)
        # print()

        state = new_state
        if reward != -2:
            game_over = True
            tmp = 0
            for a in range(100):
                tmp += testAlgo(init=1)
            episode_durations.append(tmp)
            if i % 1000 == 0:
                plot_durations()
        if step > 20:
            break
    if epsilon > 0.02:
        epsilon -= (1 / epochs)


# for i in range(20):
#     testAlgo(init=1)
