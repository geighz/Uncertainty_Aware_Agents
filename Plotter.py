import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
import os

give_labels = ('Give Advice', 'Episode', 'times as adviser')
ask_labels = ('Ask for Advice', 'Episode', 'times asked for advise')
reward_labels = ('Evaluation during Training', 'Episode', 'Reward')


def plot(x, y, title, linelabel, xlabel, ylabel, lb=None, ub=None, ylim=None):
    plt.figure(title)
    if ylim is not None:
        plt.ylim(ylim)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if ub is not None and lb is not None:
        plt.fill_between(x, ub, lb, alpha=.5)
    plt.plot(x, y, label=linelabel)
    plt.legend()


def plot_results_with_confidence_interval(linelabel, x, y, title, xlabel, ylabel, ylim=None):
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
    plot(x, averages, title, linelabel, xlabel, ylabel, ci[0::2], ci[1::2], ylim=ylim)


def save(test_results):
    if not os.path.exists('out'):
        os.mkdir('out')
    with open('out/test.results', 'wb') as config_dictionary_file:
        os.pickle.dump(test_results, config_dictionary_file)
    with open('out/test.results', 'rb') as config_dictionary_file:
        test_results = os.pickle.load(config_dictionary_file)


def save_plots():
    plt.figure(reward_labels[0])
    plt.savefig(os.path.join('out', f"{reward_labels[0]}.pdf"))
    plt.savefig(f"{reward_labels[0]}.pdf")
    plt.figure(ask_labels[0])
    plt.savefig(f"{ask_labels[0]}.pdf")
    plt.figure(give_labels[0])
    plt.savefig(f"{give_labels[0]}.pdf")


def plot_test(test_results):
    save(test_results)
    # Sort the test results by type
    test_results = test_results.values()
    agentTypes = set(map(lambda tr: tr.AgentType, test_results))
    test_results_by_setup = [[tr for tr in test_results if tr.AgentType == aT] for aT in agentTypes]

    for results in test_results_by_setup:
        label = results[0].AgentType
        epoch_ids = [test_run.EPOCH_ID for test_run in results]
        rewards = [test_run.REWARDS for test_run in results]
        times_asked = [test_run.TIMES_ASKED for test_run in results]
        times_adviser = [test_run.TIMES_GIVEN for test_run in results]

        plot_results_with_confidence_interval(label, epoch_ids, rewards, *reward_labels, ylim=(-16, 6))
        plot_results_with_confidence_interval(label, epoch_ids, times_asked, *ask_labels)
        plot_results_with_confidence_interval(label, epoch_ids, times_adviser, *give_labels)
    save_plots()
    plt.show()
