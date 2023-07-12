import numpy as np
import itertools
import pandas as pd
wall = np.array([0, 0, 1, 0, 0])
pit = np.array([0, 1, 0, 0, 0])
goal = np.array([1, 0, 0, 0, 0])


two_goal = np.zeros((506,125))
#goal
two_goal[:, 20] =  1
#pit
two_goal[:,101] = 1
player_a = np.array([0, 0, 0, 1, 0])
player_b = np.array([0, 0, 0, 0, 1])

i = 0
total_slots = [0,1,2,3,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,21,22,23,24]
perms = list(itertools.permutations(total_slots,2))
print(len(perms))

for perm in perms:
    sectionA = perm[0]*5
    sectionB = perm[1]*5
    two_goal[i,sectionA:sectionA+5] = player_a
    two_goal[i,sectionB:sectionB+5] = player_b
    i+=1

#two_goal.reshape(506)
np.savetxt("twogoal_allstatesV2.csv", two_goal, delimiter=",", fmt='%d')
#np.savetxt('twogoal_allstates.csv', two_goal)
# convert array into dataframe
