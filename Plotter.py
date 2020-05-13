import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st

give_labels = ('Giving', 'Episode', '#given advise in 25 episodes')
ask_labels = ('Ask', 'Episode', '#asked for advise in 25 episodes')
reward_labels = ('Training', 'Episode', 'Reward')


def plot(x, y, title, xlabel, ylabel, lb=None, ub=None, ylim=None, color_shading=None):
    plt.figure(title)
    # plt.clf()
    if ylim is not None:
        plt.ylim(ylim)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if ub is not None and lb is not None:
        plt.fill_between(x, ub, lb, color=color_shading, alpha=.5)
    plt.plot(x, y)
    # plt.pause(0.1)


def show_plot(title):
    plt.figure(title)
    plt.show()


def plot_histories_with_confidence_interval(x, histories, title, xlabel, ylabel, ylim=None, color_shading=None):
    x = np.stack(x)
    x = np.average(x, axis=0)

    histories = np.stack(histories)
    averages = np.average(histories, axis=0)
    standard_errors = st.sem(histories, axis=0)

    ci = np.array([])
    for index in range(len(histories[0])):
        a = histories[:, index]
        standard_error = standard_errors[index]
        average = averages[index]
        if standard_error != 0:
            # TODO: Can the confidence interval be calculated with equal areas around the median?
            # TODO: Can we assume normal distribution?
            interval = st.t.interval(0.60, len(a) - 1, loc=average, scale=standard_error)
        else:
            interval = (average, average)
        ci = np.append(ci, interval)
    plot(x, averages, title, xlabel, ylabel, ci[0::2], ci[1::2], ylim=ylim, color_shading=color_shading)


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
    plot_reward(x, reward_dic)
    plot_ask(x, asked_dic)
    plot_give(x, given_dic)
