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

test_results = test_results.values()
agentTypes = set(map(lambda x: x.AgentType, test_results))
test_results = [[tr for tr in test_results if tr.AgentType == aT] for aT in agentTypes]

for test_results_of_type in test_results:
    label = None
    epoch_ids = []
    rewards = []
    times_advisee = []
    times_adviser = []
    for test_result in test_results_of_type:
        label = test_result.AgentType
        epoch_ids.append(test_result.EPOCH_ID)
        rewards.append(test_result.REWARDS)
        times_advisee.append(test_result.TIMES_ADVISEE)
        times_adviser.append(test_result.TIMES_ADVISER)
    plot_results_with_confidence_interval(label, epoch_ids, rewards, *reward_labels, ylim=(-16, 6))
    plot_results_with_confidence_interval(label, epoch_ids, times_advisee, *ask_labels)
    plot_results_with_confidence_interval(label, epoch_ids, times_adviser, *give_labels)

plot_show()

duration = time.time() - start_time
print(f"Duration {duration} seconds")
