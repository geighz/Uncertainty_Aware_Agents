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


epochs = 1001
gamma = 0.9  # since it may take several moves to goal, making gamma high
epsilon = 1
model_a = Q_learning(80, [164, 150], 4, hidden_unit)
model_a.load_state_dict(torch.load('/Users/Lukas/repositories/Reinforcement-Learning-Q-learning-Gridworld-Pytorch/graph_output/model_a.pth'))
model_a.eval()
model_b = Q_learning(80, [164, 150], 4, hidden_unit)
model_b.load_state_dict(torch.load('/Users/Lukas/repositories/Reinforcement-Learning-Q-learning-Gridworld-Pytorch/graph_output/model_b.pth'))
model_b.eval()
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
criterion_a = torch.nn.MSELoss()
criterion_b = torch.nn.MSELoss()
buffer = 80
BATCH_SIZE = 40
memory = ReplayMemory(buffer)

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
        if np.random.random() < epsilon:  # choose random action
            action_a = np.random.randint(0, 4)
            # print("A takes random action {}".format(action_a))
        else:  # choose best action from Q(s,a) values
            action_a = np.argmax(qval_a.data)
            # print("A takes best action {}".format(action_a))
        if np.random.random() < epsilon:
            action_b = np.random.randint(0, 4)
        else:  # choose best action from Q(s,a) values
            action_b = np.argmax(qval_b.data)
        # Take action, observe new state S'
        new_state = make_move(state, action_a, action_b)
        v_new_state = Variable(torch.from_numpy(new_state)).view(1, -1)
        # Observe reward
        reward = get_reward(new_state)
        # print("reward: {}".format(reward))
        # print("New state:\n", disp_grid(new_state))
        # print("\n")
        memory.push(v_state.data, action_a, action_b, v_new_state.data, reward)
        if (len(memory) < buffer): #if buffer not filled, add to it
            state = new_state
            if reward != -2: #if reached terminal state, update game status
                break
            else:
                continue
        transitions = memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))
        state_batch = Variable(torch.cat(batch.state))
        action_a_batch = Variable(torch.LongTensor(batch.action_a)).view(-1, 1)
        action_b_batch = Variable(torch.LongTensor(batch.action_b)).view(-1, 1)
        new_state_batch = Variable(torch.cat(batch.new_state))
        reward_batch = Variable(torch.FloatTensor(batch.reward))
        non_final_mask = (reward_batch == -2)

        # tmp_state = state_batch[0].view(4,4,5).numpy()
        # print(disp_grid(tmp_state))
        # tmp_action_a = action_a_batch[0]
        # print(tmp_action_a)
        # tmp_reward = reward_batch[0]
        # print(tmp_reward)
        # tmp_new_state = new_state_batch[0].view(4,4,5).numpy()
        # print(disp_grid(tmp_new_state))

        step += 1
        qval_batch_a = model_a(state_batch)
        qval_batch_b = model_b(state_batch)
        state_action_values_a = qval_batch_a.gather(1, action_a_batch).view(1, -1)
        state_action_values_b = qval_batch_b.gather(1, action_b_batch).view(1, -1)
        newQ_a = model_a(new_state_batch)
        newQ_b = model_b(new_state_batch)
        maxQ_a = newQ_a.max(1)[0]
        maxQ_b = newQ_b.max(1)[0]
        y_a = reward_batch
        y_b = reward_batch.clone()
        y_a[non_final_mask] += gamma * maxQ_a[non_final_mask]
        y_b[non_final_mask] += gamma * maxQ_b[non_final_mask]
        y_a = y_a.view(1, -1).detach()
        y_b = y_b.view(1, -1).detach()
        loss_a = criterion_a(state_action_values_a, y_a)
        loss_b = criterion_b(state_action_values_b, y_b)

        # Optimize the model
        # Clear gradients of all optimized torch.Tensor s.
        optimizer_a.zero_grad()
        optimizer_b.zero_grad()
        # compute gradients
        loss_a.backward()
        loss_b.backward()
        # Gradient clipping can keep things stable.
        # for p in model.parameters():
        #     p.grad.data.clamp_(-1, 1)
        # update model parameters
        optimizer_a.step()
        optimizer_b.step()

        state = new_state
        if reward != -2:
            game_over = True
            tmp = 0
            for a in range(100):
                tmp += testAlgo(init=1)
            episode_durations.append(tmp)
            if i % 50 == 0:
                plot_durations()
        if step > 20:
            break
    if epsilon > 0.02:
        epsilon -= (1 / epochs)

# torch.save(model_a.state_dict(), '/Users/Lukas/repositories/Reinforcement-Learning-Q-learning-Gridworld-Pytorch/graph_output/model_a.pth')
# torch.save(model_b.state_dict(), '/Users/Lukas/repositories/Reinforcement-Learning-Q-learning-Gridworld-Pytorch/graph_output/model_b.pth')
# for i in range(20):
#     testAlgo(init=1)
