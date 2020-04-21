import matplotlib.pyplot as plt


def plot_give(x, given_dic):
    plt.figure(2)
    plt.clf()
    plt.title('Giving')
    plt.xlabel('Episode')
    plt.ylabel('#given advise in 25 episodes')
    plt.plot(x, given_dic)
    plt.pause(0.1)  # pause a bit so that plots are updated


def plot_ask(x, asked_dic):
    plt.figure(2)
    plt.clf()
    plt.title('Ask')
    plt.xlabel('Episode')
    plt.ylabel('#asked for advise in 25 episodes')
    plt.plot(x, asked_dic)
    plt.pause(0.1)


def plot_rewards(x, reward_history):
    plt.figure(2)
    plt.clf()
    plt.ylim((-16, 6))
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(x, reward_history)
    plt.pause(0.1)


def plot(x, reward_history, asked_dic, given_dic):
    plot_rewards(x, reward_history)
    plot_ask(x, asked_dic)
    plot_give(x, given_dic)
