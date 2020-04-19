import math
from DQN import ReplayMemory, Transition, hidden_unit, Body_net
from torch.autograd import Variable

from evaluation import *
from gridworld import *
from Miner import *
from Plotter import *
import torch.optim as optim
import torch

epochs = 1001
GAMMA = 0.9  # since it may take several moves to goal, making gamma high
epsilon = 1
number_heads = 4
BUFFER = 80
BATCH_SIZE = 10
TARGET_UPDATE = 5
sum_asked_for_advise = 0
sum_given_advise = 0
agent_a = Miner(number_heads)
agent_b = Miner(number_heads)
agent_a.set_partner(agent_b)
agent_b.set_partner(agent_a)
reward_history = []
x = []
asked_dic = []
given_dic = []

# TODO can I put the optimizer into the Miner class
optimizers_a = []
optimizers_b = []
criterion = torch.nn.MSELoss()
memory = ReplayMemory(BUFFER)
env = Goldmine()

# Fuer jeden head gibt es einen optimizer
for head_number in range(agent_a.policy_net.number_heads):
    optimizers_a.append(optim.Adam(agent_a.policy_net.heads[head_number].parameters()))
    optimizers_b.append(optim.Adam(agent_b.policy_net.heads[head_number].parameters()))
    # optimizers_a.append(optim.SGD(agent_a.model.heads[i].parameters(), lr=0.002))
    # optimizers_b.append(optim.SGD(agent_b.model.heads[i].parameters(), lr=0.002))

for i_episode in range(epochs):
    print("Game #: %s" % (i_episode,))
    state = env.reset()
    env.render()
    done = False
    step = 0
    # while game still in progress
    while not done:
        v_state = Variable(torch.from_numpy(state)).view(1, -1)
        # TODO: choose best action seems to return way better results
        action_a = agent_a.choose_training_action(v_state, epsilon)
        action_b = agent_b.choose_training_action(v_state, epsilon)
        # Take action, observe new state S'
        new_state, reward, done, _ = env.step(action_a, action_b)
        step += 1
        v_new_state = Variable(torch.from_numpy(new_state)).view(1, -1)
        # Observe reward
        print("reward: {}".format(reward))
        print("New state:")
        env.render()
        print("\n")
        memory.push(v_state.data, action_a, action_b, v_new_state.data, reward, not done)
        # if buffer not filled, add to it
        if len(memory) < BUFFER:
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
        non_final_mask = Variable(torch.ByteTensor(batch.non_final))
        state_action_values_a = agent_a.get_state_action_value(state_batch, action_a_batch)
        state_action_values_b = agent_b.get_state_action_value(state_batch, action_b_batch)
        # TODO: wieso haben beide agents genau die gleichen werte hier
        maxQ_a = agent_a.get_qval_for_best_action_in(new_state_batch)
        maxQ_b = agent_b.get_qval_for_best_action_in(new_state_batch)
        target_a = reward_batch
        target_b = reward_batch.clone()
        target_a[non_final_mask] += GAMMA * maxQ_a[non_final_mask]
        target_b[non_final_mask] += GAMMA * maxQ_b[non_final_mask]
        target_a = target_a.detach()
        target_b = target_b.detach()
        loss_a = []
        loss_b = []
        # TODO: hier nimmt er für jeden head den loss, eigentlich sollte der abtch nur fuer ein teil der heads verwendet werden
        for a in range(agent_a.policy_net.number_heads):
            use_sample = np.random.randint(0, agent_a.policy_net.number_heads)
            if use_sample == 0:
                loss_a.append(criterion(state_action_values_a[a].view(10), target_a))
                loss_b.append(criterion(state_action_values_b[a].view(10), target_b))
            else:
                loss_a.append(None)
                loss_b.append(None)


        # Optimize the model
        # Clear gradients of all optimized torch.Tensor s.
        for a in range(agent_a.policy_net.number_heads):
            if loss_a[a] is not None:
                # clear gradient
                optimizers_a[a].zero_grad()
                # compute gradients
                loss_a[a].backward()
                # update model parameters
                optimizers_a[a].step()
            if loss_b[a] is not None:
                optimizers_b[a].zero_grad()
                loss_b[a].backward()
                optimizers_b[a].step()
        # Gradient clipping can keep things stable.
        # for p in model.parameters():
        #     p.grad.data.clamp_(-1, 1)
        # TODO: replace with state fom state batch
        agent_a.count_state(state_batch)
        agent_b.count_state(state_batch)
        state = new_state
        if done:
            sum_asked_for_advise += agent_a.times_asked_for_advise
            sum_given_advise += agent_a.times_given_advise
            if i_episode % 25 == 0:
                x.append(i_episode)
                average_reward = evaluate_agents(agent_a, agent_b)
                reward_history.append(average_reward)
                asked_dic.append(agent_a.times_asked_for_advise)
                given_dic.append(agent_a.times_given_advise)
                agent_a.times_asked_for_advise = 0
                agent_a.times_given_advise = 0
            if i_episode % 500 == 0 and not i_episode == 0:
                plot_durations(x, reward_history)
                # plot_give(x, given_dic)
                # plot_ask(x, asked_dic)
        if step > 20:
            break
    if epsilon > 0.02:
        epsilon -= (1 / epochs)
    if i_episode % TARGET_UPDATE == 0:
        for head_number in range(agent_a.policy_net.number_heads):
            agent_a.target_net.heads[head_number].load_state_dict(agent_a.policy_net.heads[head_number].state_dict())
            agent_b.target_net.heads[head_number].load_state_dict(agent_b.policy_net.heads[head_number].state_dict())

# torch.save(model_a.state_dict(), '/Users/Lukas/repositories/Reinforcement-Learning-Q-learning-Gridworld-Pytorch/graph_output/model_a.pth')
# torch.save(model_b.state_dict(), '/Users/Lukas/repositories/Reinforcement-Learning-Q-learning-Gridworld-Pytorch/graph_output/model_b.pth')
# for i in range(20):
#     testAlgo(init=1)
