# from NoAdviceMiner import NoAdviceMiner
# from VisitBasedMiner import VisitBasedMiner
# from UncertaintyAwareMiner import UncertaintyAwareMiner
# from UncertaintyAwareMinerNormalised import UncertaintyAwareMinerNormalised
# from BayesAwareMiner import BayesAwareMiner

from SingleAwareMiner import SingleAwareMiner
# from TDMiner import TDMiner
from psutil import Process
from os import getpid
# from TestExecutor import Test_setup, execute_test
# from BayesTestExecutor import Test_setup_bayes, execute_test_bayes
from SingleTestExecutor import Test_setup_single, execute_test_single
from Plotter_single import plot_test, print_time, get_time, write_to_file, zip_out_folder
from torch.multiprocessing import Pool, Manager
from torch import multiprocessing
import os



print_time()
start_time = get_time().timestamp()
EPOCHS = 5000
BUFFER = 80
BATCH_SIZE = 10
TARGET_UPDATE =30
NUMBER_EXECUTIONS = 1
BUDGET = 100000
VA = 10000000000000.
VG = 0.0
N_AGENTS = 1
N_HEADS = 1
# loss, train,eval
test_setups = [    
    Test_setup_single(SingleAwareMiner, N_HEADS, EPOCHS, BUFFER, BATCH_SIZE, TARGET_UPDATE,BUDGET,VA,VG,'N', 'R','S',N_AGENTS),
    # Test_setup_single(Miner_Single, 1, EPOCHS, BUFFER, BATCH_SIZE, TARGET_UPDATE,'N', 'N','S'),
    # Test_setup_single(Miner_Single, 1, EPOCHS, BUFFER, BATCH_SIZE, TARGET_UPDATE,'N', 'N','R')
]

test_results = Manager().dict()

# for test_setup in test_setups:
#     for test_number in range(NUMBER_EXECUTIONS):
#         execute_test(id, test_setup, test_results)
print('Minigrid')

def limit_cpu():
    p = Process(getpid())
    # second lowest priority
    p.nice(19)


testProcesses = []
id = 0
watched_pids = 0
pool = Pool()#processes=12, initializer=limit_cpu())
context = multiprocessing.get_context('fork')
exc_tests = [execute_test_single]#[,execute_test,execute_test_bayes,execute_test_bayes]
for i,test_setup in enumerate(test_setups):
    for test_number in range(NUMBER_EXECUTIONS):
        id += 1
        watched_pids +=1
        if watched_pids> 2:
            os.wait()
            watched_pids -= 1
        testProcess = pool.Process(ctx=context, target=execute_test_single, args=(id, test_setup, test_results))
        testProcesses.append(testProcess)
        testProcess.start()

for process in testProcesses:
    process.join()

# plot_test(test_results)
# write_to_file(f"EPOCHS: {EPOCHS}", f"BUFFER: {BUFFER}", f"BATCH_SIZE: {BATCH_SIZE}",
#               f"TARGET_UPDATE: {TARGET_UPDATE}", f"NUMBER_EXECUTIONS: {NUMBER_EXECUTIONS}", f"BUDGET: {BUDGET}",
#               f"test_setups: {test_setups}")
# duration = int(get_time().timestamp() - start_time)
# print(f"Duration {duration} seconds")
# zip_out_folder()
