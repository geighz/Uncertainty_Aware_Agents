from NoAdviceMiner import NoAdviceMiner
from VisitBasedMiner import VisitBasedMiner
from UncertaintyAwareMiner import UncertaintyAwareMiner
from UncertaintyAwareMinerNormalised import UncertaintyAwareMinerNormalised
from TDMiner import TDMiner
from psutil import Process
from os import getpid
from time import strftime, time
from TestExecutor import Test_setup, execute_test
from Plotter import plot_test
from torch.multiprocessing import Pool, Manager

print(strftime("%d.%m.%Y-%H:%M:%S"))
start_time = time()
EPOCHS = 1000
BUFFER = 80
BATCH_SIZE = 10
TARGET_UPDATE = 5
NUMBER_EXECUTIONS = 10
BUDGET = 250

test_setups = [
    Test_setup(NoAdviceMiner, 1, EPOCHS, BUFFER, BATCH_SIZE, TARGET_UPDATE, BUDGET),
    Test_setup(NoAdviceMiner, 5, EPOCHS, BUFFER, BATCH_SIZE, TARGET_UPDATE, BUDGET),
    Test_setup(VisitBasedMiner, 1, EPOCHS, BUFFER, BATCH_SIZE, TARGET_UPDATE, BUDGET),
    Test_setup(VisitBasedMiner, 5, EPOCHS, BUFFER, BATCH_SIZE, TARGET_UPDATE, BUDGET),
    Test_setup(TDMiner, 5, EPOCHS, BUFFER, BATCH_SIZE, TARGET_UPDATE, BUDGET),
    Test_setup(UncertaintyAwareMiner, 5, EPOCHS, BUFFER, BATCH_SIZE, TARGET_UPDATE, BUDGET),
    Test_setup(UncertaintyAwareMinerNormalised, 5, EPOCHS, BUFFER, BATCH_SIZE, TARGET_UPDATE, BUDGET)
]

manager = Manager()
test_results = manager.dict()

# for test_setup in test_setups:
#     for test_number in range(NUMBER_EXECUTIONS):
#         execute_test(id, test_setup, test_results)


def limit_cpu():
    p = Process(getpid())
    # second lowest priority
    p.nice(19)


testProcesses = []
id = 0
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

duration = int(time() - start_time)
print(f"Duration {duration} seconds")
