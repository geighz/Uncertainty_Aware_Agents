import matplotlib.pyplot as plt
import seaborn as sns
import sys
import pickle
import os
sns.set_theme()

def load_from(folder,code):
    with open(os.path.join(folder, code, 'test.results'), 'rb') as input:
        return pickle.load(input)

# folder = sys.argv[1]
# codes = sys.argv[2:]

folder = 'out'
codes = ['20230815-044401']

experiments = {}

# relevant_data = ['AgentType','EPOCH_ID', 'REWARDS', 'TIMES_ASKED', 'TIMES_GIVEN', 'UNCERTAINTY']

relevant_data = ['AgentType','EPOCH_ID', 'REWARDS','UNCERTAINTY', 'epc','m','s']


list_of_markers = ['D-','o-','*-','v-','^-','x-']
for code in codes:
    results = load_from(folder,code)
    experiments[code] = {}
    for i,feature in enumerate(relevant_data):
        # print(results[1][i])
        experiments[code][feature] = results[1][i]
# print(experiments[code]['epc'])

plt.plot(experiments[code]['m'],label = experiments[code]['AgentType'])
plt.show()
# # print(results)
# for feature in relevant_data[2:]:
#     if feature == 'TERMINAL_TRACK_A':
#         print(experiments[code][feature][0])
#         epochs = experiments[code][feature][0]['ep']
#         means = experiments[code][feature][0]['mean']
#         std = experiments[code][feature][0]['std']
#         plt.plot(epochs,means,list_of_markers[i],label = experiments[code]['AgentType'])
#         plt.show()

#         plt.plot(epochs,std,list_of_markers[i],label = experiments[code]['AgentType'])
#         plt.show()
#     for i,code in enumerate(codes):
#         if code == 'DQN':
            
#             plt.plot(experiments[code]['EPOCH_ID'],experiments[code][feature],list_of_markers[i],label = 'DQN')#experiments[code]['AgentType'])
#         else:
#             plt.plot(experiments[code]['EPOCH_ID'],experiments[code][feature],list_of_markers[i],label = experiments[code]['AgentType'])

    
#     plt.legend()
#     plt.show()
