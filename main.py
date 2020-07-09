from NoAdviceMiner import NoAdviceMiner
from VisitBasedMiner import VisitBasedMiner
from UncertaintyAwareMiner import UncertaintyAwareMiner
from UncertaintyAwareMinerNormalised import UncertaintyAwareMinerNormalised
from TDMiner import TDMiner
from psutil import Process
from os import getpid
from TestExecutor import Test_setup, execute_test
from Plotter import plot_test, print_time, get_time
from torch.multiprocessing import Pool, Manager

print_time()
start_time = get_time().timestamp()
EPOCHS = 150000
BUFFER = 80
BATCH_SIZE = 10
TARGET_UPDATE = 5
NUMBER_EXECUTIONS = 1
BUDGET = 250

test_setups = [
    # Test_setup(NoAdviceMiner, 5, EPOCHS, BUFFER, BATCH_SIZE, TARGET_UPDATE, BUDGET, 0, 0),
    # Test_setup(VisitBasedMiner, 5, EPOCHS, BUFFER, BATCH_SIZE, TARGET_UPDATE, BUDGET, 0.9348589998939789, 2.3376882973907214),
    # Test_setup(TDMiner, 5, EPOCHS, BUFFER, BATCH_SIZE, TARGET_UPDATE, BUDGET, 1.109463093924938, 1.3107259352052203),
    Test_setup(UncertaintyAwareMiner, 5, EPOCHS, BUFFER, BATCH_SIZE, TARGET_UPDATE, 5000, 0.15314463016769106, 0.9588518836533564),
    Test_setup(UncertaintyAwareMiner, 5, EPOCHS, BUFFER, BATCH_SIZE, TARGET_UPDATE, 50000, 0.15314463016769106, 0.9588518836533564),
    Test_setup(UncertaintyAwareMiner, 5, EPOCHS, BUFFER, BATCH_SIZE, TARGET_UPDATE, 100000, 0.15314463016769106, 0.9588518836533564),
    Test_setup(UncertaintyAwareMiner, 5, EPOCHS, BUFFER, BATCH_SIZE, TARGET_UPDATE, 150000, 0.15314463016769106, 0.9588518836533564)
    # Test_setup(UncertaintyAwareMinerNormalised, 5, EPOCHS, BUFFER, BATCH_SIZE, TARGET_UPDATE, BUDGET, 1.2399574277016787, 2.330334547014672)
]

test_results = Manager().dict()

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

duration = int(get_time().timestamp() - start_time)
print(f"Duration {duration} seconds")
