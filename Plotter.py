import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st

give_labels = ('Giving', 'Episode', '#given advise in 25 episodes')
ask_labels = ('Ask', 'Episode', '#asked for advise in 25 episodes')
reward_labels = ('Training', 'Episode', 'Reward')


def plot(x, y, title, xlabel, ylabel, lb=None, ub=None, ylim=None, color_shading=None):
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


def plot_mean_with_confidence_interval(x, mean, lb, ub, ylim=None, color_shading=None):
    plot(x, mean, *reward_labels, lb=lb, ub=ub, ylim=ylim, color_shading=color_shading)


def plot_histories_with_confidence_interval(x, histories, ylim=None, color_shading=None):
    x = np.stack(x)
    x = np.average(x, axis=0)

    histories = np.stack(histories)
    avg = np.average(histories, axis=0)

    ci = np.array([])
    for data_point_number in range(len(histories[0])):
        a = histories[:, data_point_number]
        standard_error = st.sem(a)
        mean = np.mean(a)
        if standard_error != 0:
            interval = st.t.interval(0.60, len(a) - 1, loc=mean, scale=standard_error)
        else:
            interval = (mean, mean)
        ci = np.append(ci, interval)
    plot_mean_with_confidence_interval(x, avg, ci[0::2], ci[1::2], ylim=ylim, color_shading=color_shading)


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
