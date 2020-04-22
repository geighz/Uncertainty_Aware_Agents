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
    x = x.tolist()
    reward_history = reward_history.tolist()
    asked_dic = asked_dic.tolist()
    given_dic = given_dic.tolist()
    plot_rewards(x, reward_history)
    plot_ask(x, asked_dic)
    plot_give(x, given_dic)


def plot_mean_and_CI(x, mean, lb, ub, color_mean=None, color_shading=None):
    plt.figure(2)
    plt.clf()
    plt.ylim((-16, 6))
    # plot the shaded range of the confidence intervals
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.fill_between(x, ub, lb, color=color_shading, alpha=.5)
    # plot the mean on top
    plt.plot(x, mean)
    plt.pause(0.1)
