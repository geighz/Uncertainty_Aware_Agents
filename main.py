from NoAdviceMiner import NoAdviceMiner
from VisitBasedMiner import VisitBasedMiner
from UncertaintyAwareMiner import UncertaintyAwareMiner
from UncertaintyAwareMinerNormalised import UncertaintyAwareMinerNormalised
from TestExecutor import *
from Plotter import *
from torch.multiprocessing import Pool, Manager
import psutil
import os
import time

start_time = time.time()
EPOCHS = 350000
BUFFER = 80
BATCH_SIZE = 10
TARGET_UPDATE = 5
NUMBER_EXECUTIONS = 1
BUDGET = 250

test_setups = [
    Test_setup(NoAdviceMiner, 1, EPOCHS, BUFFER, BATCH_SIZE, TARGET_UPDATE, BUDGET),
    Test_setup(VisitBasedMiner, 1, EPOCHS, BUFFER, BATCH_SIZE, TARGET_UPDATE, BUDGET),
    Test_setup(UncertaintyAwareMiner, 5, EPOCHS, BUFFER, BATCH_SIZE, TARGET_UPDATE, BUDGET),
    Test_setup(UncertaintyAwareMinerNormalised, 5, EPOCHS, BUFFER, BATCH_SIZE, TARGET_UPDATE, BUDGET)
]

manager = Manager()
test_results = manager.dict()

# for test_setup in test_setups:
#     for test_number in range(NUMBER_EXECUTIONS):
#         execute_test(id, test_setup, test_results)


def limit_cpu():
    p = psutil.Process(os.getpid())
    # second lowest priority
    p.nice(19)


testProcesses = []
id = 0
print(psutil.cpu_count(logical=False))
print(psutil.cpu_count(logical=True))
pool = Pool(processes=12, initializer=limit_cpu())
for test_setup in test_setups:
    for test_number in range(NUMBER_EXECUTIONS):
        id += 1
        testProcess = pool.Process(target=execute_test, args=(id, test_setup, test_results))
        testProcesses.append(testProcess)
        testProcess.start()

for process in testProcesses:
    process.join()

plot_test(test_results)

duration = int(time.time() - start_time)
print(f"Duration {duration} seconds")
