from TestExecutor import *
from Plotter import *
import multiprocessing
import time

from NoAdviceMiner import NoAdviceMiner
from VisitBasedMiner import VisitBasedMiner
from UncertaintyAwareMiner import UncertaintyAwareMiner
from UncertaintyAwareMinerNormalised import UncertaintyAwareMinerNormalised

start_time = time.time()
EPOCHS = 1001
BUFFER = 80
BATCH_SIZE = 10
TARGET_UPDATE = 5
NUMBER_EXECUTIONS = 1

test_setups = [
    # Test_setup(NoAdviceMiner, 1, EPOCHS, BUFFER, BATCH_SIZE, TARGET_UPDATE),
    # Test_setup(VisitBasedMiner, 1, EPOCHS, BUFFER, BATCH_SIZE, TARGET_UPDATE),
    # Test_setup(UncertaintyAwareMiner, 5, EPOCHS, BUFFER, BATCH_SIZE, TARGET_UPDATE),
    Test_setup(UncertaintyAwareMinerNormalised, 5, EPOCHS, BUFFER, BATCH_SIZE, TARGET_UPDATE)
]

manager = multiprocessing.Manager()
test_results = manager.dict()
testProcesses = []
id = 0
for test_setup, test_number in (test_setups, range(NUMBER_EXECUTIONS)):
    id += 1
    testProcess = multiprocessing.Process(target=execute_test, args=(test_setup, id, test_results))
    testProcesses.append(testProcess)
    testProcess.start()

for process in testProcesses:
    process.join()

plot(test_results)

duration = int(time.time() - start_time)
print(f"Duration {duration} seconds")
