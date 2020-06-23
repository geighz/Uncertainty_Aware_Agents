from TestExecutor import *
from VisitBasedMiner import VisitBasedMiner
from Plotter import *
import multiprocessing
import time

start_time = time.time()
EPOCHS = 1001
BUFFER = 80
BATCH_SIZE = 10
TARGET_UPDATE = 5
NUMBER_EXECUTIONS = 1

test_setups = [#Test_setup(VisitBasedMiner, 1, EPOCHS, BUFFER, BATCH_SIZE, TARGET_UPDATE),
               Test_setup(UncertaintyAwareMiner, 5, EPOCHS, BUFFER, BATCH_SIZE, TARGET_UPDATE),
               Test_setup(UncertaintyAwareMinerNormalised, 5, EPOCHS, BUFFER, BATCH_SIZE, TARGET_UPDATE)]
               #Test_setup(NoAdviceMiner, 1, EPOCHS, BUFFER, BATCH_SIZE, TARGET_UPDATE)]

manager = multiprocessing.Manager()
test_results = manager.dict()
testProcesses = []
id = 0
for test_setup in test_setups:
    for test_number in range(NUMBER_EXECUTIONS):
        id += 1
        testProcess = multiprocessing.Process(target=execute_test, args=(test_setup, id, test_results))
        testProcesses.append(testProcess)
        testProcess.start()

for process in testProcesses:
    process.join()

agentTypes = set(map(lambda x: x[0], test_results.values()))
test_results = [[y for y in test_results.values() if y[0] == x] for x in agentTypes]

for test_result in test_results:
    label = None
    epoch_ids = []
    rewards = []
    times_advisee = []
    times_adviser = []
    for tmp in test_result:
        label = tmp[0]
        epoch_ids.append(tmp[1])
        rewards.append(tmp[2])
        times_advisee.append(tmp[3])
        times_adviser.append(tmp[4])
    plot_results_with_confidence_interval(label, epoch_ids, rewards, *reward_labels, ylim=(-16, 6))
    plot_results_with_confidence_interval(label, epoch_ids, times_advisee, *ask_labels)
    plot_results_with_confidence_interval(label, epoch_ids, times_adviser, *give_labels)

plot_show()

duration = time.time() - start_time
print(f"Duration {duration} seconds")
