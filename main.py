from TestExecuter import *
from Plotter import *

EPOCHS = 501
BUFFER = 80
BATCH_SIZE = 10
TARGET_UPDATE = 5
NUMBER_EXECUTIONS = 3

test_setups = [Test_setup(VisitBasedMiner, 1, EPOCHS, BUFFER, BATCH_SIZE, TARGET_UPDATE),
               Test_setup(UncertaintyAwareMiner, 5, EPOCHS, BUFFER, BATCH_SIZE, TARGET_UPDATE)]
test_results = []

for test in test_setups:
    test_results.append(execute_test(test, NUMBER_EXECUTIONS))

for test_result in test_results:
    label, epoch_ids, rewards, times_advisee, times_adviser = test_result
    plot_histories_with_confidence_interval(label, epoch_ids, rewards, *reward_labels, ylim=(-16, 6))
    plot_histories_with_confidence_interval(label, epoch_ids, times_advisee, *ask_labels)
    plot_histories_with_confidence_interval(label, epoch_ids, times_adviser, *give_labels)

plot_show()
