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


def track_progress(episode_number):
    if i_episode % 25 == 0:
        x.append(episode_number)
        average_reward = evaluate_agents(agent_a, agent_b)
        reward_history.append(average_reward)
        asked_dic.append(agent_a.times_advisee)
        given_dic.append(agent_a.times_adviser)
        agent_a.times_advisee = 0
        agent_a.times_adviser = 0
    if episode_number % 1000 == 0 and not episode_number == 0:
        plot(x, reward_history, asked_dic, given_dic)


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
            track_progress(i_episode)
        if step > 20:
            break
    if epsilon > 0.02:
        epsilon -= (1 / epochs)
    if i_episode % TARGET_UPDATE == 0:
        for head_number in range(agent_a.policy_net.number_heads):
            agent_a.set_target_to_policy_net()
            agent_b.set_target_to_policy_net()
