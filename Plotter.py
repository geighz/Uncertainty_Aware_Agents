import pickle
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
import os
import matplotlib
import matplotlib.cm as cmx
from mpl_toolkits.mplot3d import Axes3D
from time import strftime
import pytz
from datetime import datetime

uncertainty_labels = ('Average Uncertainty', 'Episode', 'Uncertainty \u03BC', (0, 3))
give_labels = ('Give_Advice', 'Episode', 'times as adviser')
ask_labels = ('Ask_for_Advice', 'Episode', 'times asked for advise')
reward_labels = ('Evaluation_during Training', 'Episode', 'Reward')
start_time = strftime("%Y%m%d-%H%M%S")
out_folder = os.path.join('out', start_time)
timezone_berlin = pytz.timezone('Europe/Berlin')
date_format = '%d.%m.%Y-%H:%M:%S %Z%z'
start_time_str = datetime.now(timezone_berlin).strftime(date_format)


def plot(x, y, title, linelabel, xlabel, ylabel, lb=None, ub=None, ylim=None):
    plt.figure(title)
    # if ylim is not None:
    #     plt.ylim(ylim)
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
    y[y == 0] = np.nan
    mean = np.nanmean(y, axis=0)
    mean[np.isnan(mean)] = 0
    standard_errors = st.sem(y, axis=0, nan_policy='omit')
    ci = np.array([])
    for index in range(len(y[0])):
        a = y[:, index]
        standard_error = standard_errors[index]
        average = mean[index]
        if standard_error != 0:
            # TODO: Can the confidence interval be calculated with equal areas around the median?
            # TODO: Can we assume normal distribution?
            interval = st.t.interval(0.60, len(a) - 1, loc=average, scale=standard_error)
        else:
            interval = (average, average)
        ci = np.append(ci, interval)
    plot(x, mean, title, linelabel, xlabel, ylabel, ci[0::2], ci[1::2], ylim=ylim)


def create_folder():
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)


def save_plots():
    plt.figure(reward_labels[0])
    plt.savefig(os.path.join(out_folder, f"{reward_labels[0]}.pdf"))
    plt.figure(ask_labels[0])
    plt.savefig(os.path.join(out_folder, f"{ask_labels[0]}.pdf"))
    plt.figure(give_labels[0])
    plt.savefig(os.path.join(out_folder, f"{give_labels[0]}.pdf"))
    plt.figure(uncertainty_labels[0])
    plt.savefig(os.path.join(out_folder, f"{uncertainty_labels[0]}.pdf"))


def save(test_results):
    with open(os.path.join(out_folder, 'test.results'), 'wb') as input:
        pickle.dump(dict(test_results), input)


def load_from(date_time):
    with open(os.path.join('out', date_time, 'test.results'), 'rb') as input:
        return pickle.load(input)

    
def plot_test(test_results):
    create_folder()
    save(test_results)
    # test_results = load_from('20200626-164138')
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
        uncertainty = [test_run.UNCERTAINTY for test_run in results]

        #plot_results_with_confidence_interval(label, epoch_ids, rewards, *reward_labels, ylim=(-16, 6))
        plot_results_with_confidence_interval(label, epoch_ids, rewards, *reward_labels)
        plot_results_with_confidence_interval(label, epoch_ids, times_asked, *ask_labels)
        plot_results_with_confidence_interval(label, epoch_ids, times_adviser, *give_labels)
        plot_results_with_confidence_interval(label, epoch_ids, uncertainty, *uncertainty_labels)
    create_folder()
    save_plots()
    plt.show()


def get_scalar_map(cs, colors_map='jet'):
    cmap = plt.get_cmap(colors_map)
    cNorm = matplotlib.colors.Normalize(vmin=min(cs), vmax=max(cs))
    scalar_map = cmx.ScalarMappable(norm=cNorm, cmap=cmap)
    return scalar_map


def scatter3d(x, y, z):
    cs = np.asarray(z)
    fig = plt.figure()
    ax = Axes3D(fig)
    scalar_map = get_scalar_map(cs)
    ax.scatter(x, y, z, c=scalar_map.to_rgba(cs))
    scalar_map.set_array(cs)
    fig.colorbar(scalar_map)
    ax.set_xlabel('va')
    ax.set_ylabel('vg')
    ax.set_zlabel('reward')
    create_folder()
    plt.savefig(os.path.join(out_folder, "scatter3D.pdf"))
    plt.show()


def scatter2d(x, y, z):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cs = np.asarray(z)
    scalar_map = get_scalar_map(cs)
    ax.scatter(x, y, c=scalar_map.to_rgba(cs), marker='o')
    ax.set_xlabel('va')
    ax.set_ylabel('vg')
    create_folder()
    plt.savefig(os.path.join(out_folder, "scatter2D.pdf"))
    plt.show()


def write_to_file(*args):
    text = '\n'.join(str(x) for x in args)
    stream = os.popen('git rev-parse --verify HEAD')
    git_hash = stream.read()
    end_time_str = print_time()
    text = '\n'.join(str(x) for x in [text, git_hash, start_time_str, end_time_str])
    create_folder()
    file = open(os.path.join(out_folder, "Test_notes.txt"), mode="w", encoding="utf-8")
    file.write(text)
    file.close()


def zip_out_folder():
    stream = os.popen(f'zip -r out/{start_time} {out_folder}')
    output = stream.read()
    print(output)


def get_time():
    return datetime.now(timezone_berlin)


def print_time():
    time_str = get_time().strftime(date_format)
    print(time_str)
    return time_str
