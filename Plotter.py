import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st

give_labels = ('Giving', 'Episode', '#given advise in 25 episodes')
ask_labels = ('Ask', 'Episode', '#asked for advise in 25 episodes')
reward_labels = ('Training', 'Episode', 'Reward')


def plot(x, y, title, linelabel, xlabel, ylabel, lb=None, ub=None, ylim=None, color_shading=None):
    plt.figure(title)
    if ylim is not None:
        plt.ylim(ylim)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if ub is not None and lb is not None:
        plt.fill_between(x, ub, lb, color=color_shading, alpha=.5)
    plt.plot(x, y, label=linelabel)
    plt.legend()


def plot_show():
    plt.show()


def plot_histories_with_confidence_interval(linelabel, x, y, title, xlabel, ylabel, ylim=None, color_shading=None):
    x = np.stack(x)
    x = np.average(x, axis=0)

    y = np.stack(y)
    averages = np.average(y, axis=0)
    standard_errors = st.sem(y, axis=0)

    ci = np.array([])
    for index in range(len(y[0])):
        a = y[:, index]
        standard_error = standard_errors[index]
        average = averages[index]
        if standard_error != 0:
            # TODO: Can the confidence interval be calculated with equal areas around the median?
            # TODO: Can we assume normal distribution?
            interval = st.t.interval(0.60, len(a) - 1, loc=average, scale=standard_error)
        else:
            interval = (average, average)
        ci = np.append(ci, interval)
    plot(x, averages, title, linelabel, xlabel, ylabel, ci[0::2], ci[1::2], ylim=ylim, color_shading=color_shading)

