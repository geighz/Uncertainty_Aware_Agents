from TestExecutor import *
from Plotter import *
import multiprocessing
import time

start_time = time.time()
EPOCHS = 251
BUFFER = 80
BATCH_SIZE = 10
TARGET_UPDATE = 5
NUMBER_EXECUTIONS = 1

test_setups = [Test_setup(VisitBasedMiner, 1, EPOCHS, BUFFER, BATCH_SIZE, TARGET_UPDATE),
               Test_setup(UncertaintyAwareMiner, 5, EPOCHS, BUFFER, BATCH_SIZE, TARGET_UPDATE),
               Test_setup(NoAdviceMiner, 1, EPOCHS, BUFFER, BATCH_SIZE, TARGET_UPDATE)]
test_results = []

def a_func(t):
  test_results.append(execute_test(t, NUMBER_EXECUTIONS))

#for test in test_setups:
    #test_results.append(execute_test(test, NUMBER_EXECUTIONS))

p=multiprocessing.Pool(3)
p.map(a_func, test_setups)
#p.join()

#with multiprocessing.Pool() as pool:
    #pool.map(a_func, test_setups)
#reader_process = multiprocessing.Process(target=a_func, args=test_setups)
#reader_process.start()
#reader_process.join()

for test_result in test_results:
    label, epoch_ids, rewards, times_advisee, times_adviser = test_result
    plot_results_with_confidence_interval(label, epoch_ids, rewards, *reward_labels, ylim=(-16, 6))
    plot_results_with_confidence_interval(label, epoch_ids, times_advisee, *ask_labels)
    plot_results_with_confidence_interval(label, epoch_ids, times_adviser, *give_labels)

plot_show()

duration = time.time() - start_time
print(f"Duration {duration} seconds")