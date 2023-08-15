from NoAdviceMiner import NoAdviceMiner
from VisitBasedMiner import VisitBasedMiner
from UncertaintyAwareMiner import UncertaintyAwareMiner
from UncertaintyAwareMinerNormalised import UncertaintyAwareMinerNormalised
from BayesAwareMiner import BayesAwareMiner
from TDMiner import TDMiner
from psutil import Process
from os import getpid
from TestExecutor import Test_setup, execute_test
from BayesTestExecutor import Test_setup_bayes, execute_test_bayes
from Plotter import plot_test, print_time, get_time, write_to_file, zip_out_folder
from torch.multiprocessing import Pool, Manager
from torch import multiprocessing
import time
import os




print_time()
start_time = get_time().timestamp()
EPOCHS = 30_000
BUFFER = 80
BATCH_SIZE = 10
TARGET_UPDATE =30
NUMBER_EXECUTIONS = 6
BUDGET = 100000
# loss, train,eval
test_setups = [
     Test_setup(NoAdviceMiner, 1, EPOCHS, BUFFER, BATCH_SIZE, TARGET_UPDATE, BUDGET, 0, 0),
    # Test_setup(VisitBasedMiner, 5, EPOCHS, BUFFER, BATCH_SIZE, TARGET_UPDATE, BUDGET, 1.41, 2.2),
    # Test_setup(TDMiner, 5, EPOCHS, BUFFER, BATCH_SIZE, TARGET_UPDATE, BUDGET, 1.21, 1.51),
    # Test_setup(UncertaintyAwareMiner, 5, EPOCHS, BUFFER, BATCH_SIZE, TARGET_UPDATE, BUDGET, 0.14, 1.61),
    # Test_setup(UncertaintyAwareMinerNormalised, 5, EPOCHS, BUFFER, BATCH_SIZE, TARGET_UPDATE, BUDGET, 0.77, 2.45),
    # loss, train,eval
    Test_setup_bayes(BayesAwareMiner, 1, EPOCHS, BUFFER, BATCH_SIZE, TARGET_UPDATE, BUDGET, 0., 0.0,'N', 'N','N'),
    # Test_setup_bayes(BayesAwareMiner, 1, EPOCHS, BUFFER, BATCH_SIZE, TARGET_UPDATE, BUDGET, 0., 0.0,'N', 'S','N'),
    # Test_setup_bayes(BayesAwareMiner, 1, EPOCHS, BUFFER, BATCH_SIZE, TARGET_UPDATE, BUDGET, 0., 0.0,'N', 'S','S'),
    # Test_setup_bayes(BayesAwareMiner, 1, EPOCHS, BUFFER, BATCH_SIZE, TARGET_UPDATE, BUDGET, 0., 0.0,'N', 'R','N'),
    # Test_setup_bayes(BayesAwareMiner, 1, EPOCHS, BUFFER, BATCH_SIZE, TARGET_UPDATE, BUDGET, 0., 0.0,'N', 'R','R'),
    # Test_setup_bayes(BayesAwareMiner, 1, EPOCHS, BUFFER, BATCH_SIZE, TARGET_UPDATE, BUDGET, 0., 0.0,'S', 'S','S'),
    # Test_setup_bayes(BayesAwareMiner, 1, EPOCHS, BUFFER, BATCH_SIZE, TARGET_UPDATE, BUDGET, 0., 0.0,'R', 'R','R')
    # Test_setup_bayes(BayesAwareMiner, 1, EPOCHS, BUFFER, BATCH_SIZE, TARGET_UPDATE, BUDGET, 0., 0.0,'R', 'N','N'),
    # Test_setup_bayes(BayesAwareMiner, 1, EPOCHS, BUFFER, BATCH_SIZE, TARGET_UPDATE, BUDGET, 0., 0.0,'S', 'N','N')

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
watched_pids = []
pool = Pool()#processes=12, initializer=limit_cpu())
context = multiprocessing.get_context('fork')
#exc_tests = [execute_test_bayes,execute_test_bayes,execute_test_bayes,execute_test_bayes,execute_test_bayes,execute_test_bayes]#[,execute_test,execute_test_bayes,execute_test_bayes]
for i,test_setup in enumerate(test_setups):
    if test_setup[0] == BayesAwareMiner:
        exc_tests = execute_test_bayes
    else:
        exc_tests = execute_test
    for test_number in range(NUMBER_EXECUTIONS):
        id += 1
        watched_pids.append(id)
        if len(watched_pids) > 2:
            print(id)
            os.wait()
            watched_pids.pop(0)
        testProcess = pool.Process(ctx=context, target=exc_tests, args=(id, test_setup, test_results))
        testProcesses.append(testProcess)
        testProcess.start()


# for pid in watched_pids:
#     os.wait()


for process in testProcesses:
    process.join()

plot_test(test_results)
write_to_file(f"EPOCHS: {EPOCHS}", f"BUFFER: {BUFFER}", f"BATCH_SIZE: {BATCH_SIZE}",
              f"TARGET_UPDATE: {TARGET_UPDATE}", f"NUMBER_EXECUTIONS: {NUMBER_EXECUTIONS}", f"BUDGET: {BUDGET}",
              f"test_setups: {test_setups}")
duration = int(get_time().timestamp() - start_time)
print(f"Duration {duration} seconds")
zip_out_folder()
