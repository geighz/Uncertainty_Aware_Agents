from evaluation import *
from Miner import *
from Plotter import *
import torch

epochs = 1001
epsilon = 1
number_heads = 4
BUFFER = 80
BATCH_SIZE = 10
TARGET_UPDATE = 5
agent_a = Miner(number_heads)
agent_b = Miner(number_heads)
agent_a.set_partner(agent_b)
agent_b.set_partner(agent_a)
reward_history = []
x = []
asked_dic = []
given_dic = []

memory = ReplayMemory(BUFFER)
env = Goldmine()

for i_episode in range(epochs):
    print("Game #: %s" % (i_episode,))
    env.reset()
    done = False
    step = 0
    # while game still in progress
    while not done:
        v_state = env.v_state
        action_a = agent_a.choose_training_action(env, epsilon)
        action_b = agent_b.choose_training_action(env, epsilon)
        # Take action, observe new state S'
        _, reward, done, _ = env.step(action_a, action_b)
        step += 1
        memory.push(v_state.data, action_a, action_b, env.v_state.data, reward, not done)
        # if buffer not filled, add to it
        if len(memory) < BUFFER:
            if done:
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
        agent_a.optimize(state_batch, action_a_batch, new_state_batch, reward_batch, non_final_mask)
        agent_b.optimize(state_batch, action_b_batch, new_state_batch, reward_batch, non_final_mask)
        if done:
            if i_episode % 25 == 0:
                x.append(i_episode)
                average_reward = evaluate_agents(agent_a, agent_b)
                reward_history.append(average_reward)
                asked_dic.append(agent_a.times_advisee)
                given_dic.append(agent_a.times_advisor)
                agent_a.times_advisee = 0
                agent_a.times_advisor = 0
            if i_episode % 500 == 0 and not i_episode == 0:
                plot_rewards(x, reward_history)
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
