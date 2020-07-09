import optuna
from NoAdviceMiner import NoAdviceMiner
from VisitBasedMiner import VisitBasedMiner
from UncertaintyAwareMiner import UncertaintyAwareMiner
from UncertaintyAwareMinerNormalised import UncertaintyAwareMinerNormalised
from TDMiner import TDMiner
from time import strftime, time
from TestExecutor import Test_setup, execute_test
from torch.multiprocessing import Manager
from Plotter import scatter2d, scatter3d, write_to_file, get_time, print_time
import json

EPOCHS = 400
BUFFER = 80
BATCH_SIZE = 10
TARGET_UPDATE = 5
NUMBER_EXECUTIONS = 9
n_trials = 100
BUDGET = 250
miner = UncertaintyAwareMiner
x = []
y = []
z = []

start_time = get_time().timestamp()
start_time_str = print_time()


def run(test_setup):
    test_results = Manager().dict()
    execute_test(0, test_setup, test_results)
    result = 0
    for ev in test_results[0].REWARDS:
        result += ev
    result -= test_results[0].REWARDS[0]
    print('result:', result)
    return result


def objective(trial):
    va = trial.suggest_uniform('va', 0, 3)
    vg = trial.suggest_uniform('vg', 0, 3)
    test_setup = Test_setup(miner, 5, EPOCHS, BUFFER, BATCH_SIZE, TARGET_UPDATE, BUDGET, va, vg)
    results = []
    for i in range(NUMBER_EXECUTIONS):
        results.append(run(test_setup))
    results = sorted(results)
    result = 0
    lb = int(len(results) / 3)
    ub = len(results) - lb
    for i in range(ub - lb):
        result += results[i + lb]
    result /= ub - lb
    x.append(va)
    y.append(vg)
    z.append(result)
    return result


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=n_trials)

print(study.best_params)
print(study.best_value)
print(study.best_trial)
print(len(study.trials))

end_time = get_time().timestamp()
end_time_str = print_time()
write_to_file(miner.__name__, f"NUMBER_EXECUTIONS: {NUMBER_EXECUTIONS}", f"n_trials: {n_trials}",
              study.best_params, study.best_value, study.best_trial, start_time_str, end_time_str)
scatter2d(x, y, z)
scatter3d(x, y, z)

duration = int(end_time - start_time)
print(f"Duration {duration} seconds")
