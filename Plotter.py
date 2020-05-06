import matplotlib.pyplot as plt

give_labels = ('Giving', 'Episode', '#given advise in 25 episodes')
ask_labels = ('Ask', 'Episode', '#asked for advise in 25 episodes')
reward_labels = ('Training', 'Episode', 'Reward')


def plot(x, y, title, xlabel, ylabel, ylim=None, lb=None, ub=None, color_shading=None):
    plt.figure(1)
    plt.clf()
    if ylim is not None:
        plt.ylim(ylim)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if ub is not None and lb is not None:
        plt.fill_between(x, ub, lb, color=color_shading, alpha=.5)
    plt.plot(x, y)
    plt.pause(0.1)


def plot_with_confidence_interval(x, mean, lb, ub, color_shading=None):
    plot(x, mean, *reward_labels, ylim=(-16, 6), lb=lb, ub=ub, color_shading=color_shading)


def plot_give(x, given_dic):
    plot(x, given_dic, *give_labels)


def plot_ask(x, asked_dic):
    plot(x, asked_dic, *ask_labels)


def plot_reward(x, reward_dic):
    plot(x, reward_dic, *reward_labels, ylim=(-16, 6))


def plot_rew_ask_giv(x, reward_dic, asked_dic, given_dic):
    x = x.tolist()
    reward_dic = reward_dic.tolist()
    asked_dic = asked_dic.tolist()
    given_dic = given_dic.tolist()
    # plot_reward(x, reward_dic)
    plot_ask(x, asked_dic)
    plot_give(x, given_dic)
