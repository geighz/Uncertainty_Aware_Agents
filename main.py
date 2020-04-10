import math
from DQN import ReplayMemory, Transition, hidden_unit, Q_learning
from torch.autograd import Variable
from gridworld import *
from Miner import *
from Plotter import *
import torch.optim as optim
import torch

reward_history = []


# Here is the test of AI
def test_algo(init=0):
    if init == 0:
        state = init_grid()
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
        action_a = agent_a.choose_best_action(state)
        action_b = agent_b.choose_best_action(state)
        # print('A: Move #: %s; Taking action: %s' % (i, action_a))
        # print('B: Move #: %s; Taking action: %s' % (i, action_b))
        state = make_move(state, action_a, action_b)
        # print(disp_grid(state))
        reward = get_reward(state)
        reward_sum += reward
        if is_done(state):
            status = 0
            return reward_sum
            # print("Reward: %s" % (reward,))
        i += 1  # If we're taking more than 10 actions, just stop, we probably can't win this game
        if i > 10:
            # print("Reward: Game lost; too many moves.")
            return reward_sum


# Here is the test of AI
def test_all_states():
    reward_sum = 0
    for game_id in range(50):
        state = load_state_with_id(game_id)
        steps = 0
        status = 1
        while status == 1:
            action_a = agent_a.choose_best_action(state)
            action_b = agent_b.choose_best_action(state)
            state = make_move(state, action_a, action_b)
            reward = get_reward(state)
            reward_sum += reward
            steps += 1
            if is_done(state) or steps > 10:
                status = 0
    return reward_sum/50

epochs = 501
gamma = 0.9  # since it may take several moves to goal, making gamma high
epsilon = 1

# for hidden in model.hidden_units:
#     print(hidden.nn.weight.size())
#     print(hidden.nn.weight)
# print(model.final_unit.weight.size())
# print(model.final_unit.weight)
# optimizer_a = optim.RMSprop(model_a.parameters(), lr=0.001)
# optimizer_b = optim.RMSprop(model_b.parameters(), lr=0.001)
agent_a = Miner()
agent_b = Miner()
agent_a.set_partner(agent_b)
agent_b.set_partner(agent_a)

# TODO can I put the optimizer into the Miner class
optimizer_a = optim.Adam(agent_a.get_model_parameters(), lr=0.001)
optimizer_b = optim.Adam(agent_b.get_model_parameters(), lr=0.001)
# optimizer_a = optim.SGD(model_a.parameters(), lr=0.02)
# optimizer_b = optim.SGD(model_b.parameters(), lr=0.02)
criterion_a = torch.nn.MSELoss()
criterion_b = torch.nn.MSELoss()
buffer = 1
BATCH_SIZE = 1
memory = ReplayMemory(buffer)
sum_asked_for_advise = 0
x = []
asked_dic = []
sum_given_advise = 0
given_dic = []
for i in range(epochs):
    print("Game #: %s" % (i,))
    state = init_grid_player()
    print(render(state))
    game_over = False
    step = 0
    # while game still in progress
    while not game_over:
        v_state = Variable(torch.from_numpy(state)).view(1, -1)

        # TODO: choose best action seems to return way better results
        action_a = agent_a.choose_training_action(state, epsilon)
        action_b = agent_b.choose_training_action(state, epsilon)
        # Take action, observe new state S'
        new_state = make_move(state, action_a, action_b)
        v_new_state = Variable(torch.from_numpy(new_state)).view(1, -1)
        # Observe reward
        reward = get_reward(new_state)
        print("reward: {}".format(reward))
        print("New state:\n", render(new_state))
        print("\n")
        memory.push(v_state.data, action_a, action_b, v_new_state.data, reward)
        # if buffer not filled, add to it
        if len(memory) < buffer:
            state = new_state
            if reward != -2:
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
        step += 1
        state_action_values_a = agent_a.get_state_action_value(state_batch, action_a_batch)
        state_action_values_b = agent_b.get_state_action_value(state_batch, action_b_batch)
        # TODO: wieso haben beide agents genau die gleichen werte hier
        maxQ_a = agent_a.get_max_q_value(new_state_batch)
        maxQ_b = agent_b.get_max_q_value(new_state_batch)
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
        # TODO: replace with state fom state batch
        agent_a.count_state(state)
        agent_b.count_state(state)
        state = new_state
        if is_done(state):
            game_over = True
            sum_asked_for_advise += agent_a.times_asked_for_advise
            sum_given_advise += agent_a.times_given_advise
            if i % 25 == 0:
                x.append(i)
                average_reward = test_all_states()
                reward_history.append(average_reward)
                asked_dic.append(agent_a.times_asked_for_advise)
                given_dic.append(agent_a.times_given_advise)
                agent_a.times_asked_for_advise = 0
                agent_a.times_given_advise = 0
            if i % 500 == 0 and not i == 0:
                plot_durations(x, reward_history)
                plot_give(x, given_dic)
                plot_ask(x, asked_dic)
        if step > 20:
            break
    if epsilon > 0.02:
        epsilon -= (1 / epochs)

# torch.save(model_a.state_dict(), '/Users/Lukas/repositories/Reinforcement-Learning-Q-learning-Gridworld-Pytorch/graph_output/model_a.pth')
# torch.save(model_b.state_dict(), '/Users/Lukas/repositories/Reinforcement-Learning-Q-learning-Gridworld-Pytorch/graph_output/model_b.pth')
# for i in range(20):
#     testAlgo(init=1)
