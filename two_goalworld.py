import numpy as np
import torch
from torch.autograd import Variable
import pickle
import os.path

player_a = np.array([0, 0, 0, 1, 0])
player_b = np.array([0, 0, 0, 0, 1])
wall = np.array([0, 0, 1, 0, 0])
pit = np.array([0, 1, 0, 0, 0])
goal = np.array([1, 0, 0, 0, 0])
all_states = np.genfromtxt('twogoal_allstatesV2.csv', delimiter=',')
dictionary_states = {}
number_of_eval_games =506
state_size = 125

for i in range(506):
    dictionary_states[i] = all_states[i].reshape((5, 5, 5))





class TwoGoal:
    def __init__(self):
        self.state = init_grid_player()
        self.isDone = False
        self.deterministic = True
        
        if ~os.path.isfile('all_states_two_goal.pickle'):
            all_states = np.genfromtxt('twogoal_allstatesV2.csv', delimiter=',')
            for i in range(506):
                all_states[i].reshape((5, 5, 5))
            with open('all_states_two_goal.pickle', 'wb') as handle:
                pickle.dump(all_states, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open('all_states_two_goal.pickle', 'r') as handle:
                all_states = pickle.load(handle)


    def step(self, action_a, action_b):
        if not self.isDone:
            self.state = make_move(self.state, action_a, action_b)
            self.isDone = is_done(self.state)
            self.update_v_state()
        return self.state, get_reward(self.state), self.isDone, "Info"

    def reset(self, state_id=None):
        if state_id is None:
            self.state = init_grid_rand()
        else:
            self.state = load_state_with_id(state_id)
            #render(self.state)
        self.update_v_state()
        self.isDone = False
        return self.state

    def render(self, mode='human'):
        render(self.state)

    def render_state(self, state):
        render(state.reshape((5,5,5)).detach().numpy())

    def update_v_state(self):
        self.v_state = v_state(self.state)

    def save_states(self, np_array):
        np.savetxt("states.csv", np_array, delimiter=",", fmt='%d')


def v_state(state):
    
    return Variable(torch.from_numpy(state)).view(1, -1)

def load_state_with_id(state_id):
    # print(state_id)
   
    return all_states[state_id].reshape((5, 5, 5))

def save_dictionary():

    with open('all_states.pickle', 'wb') as handle:
        pickle.dump(all_states, handle, protocol=pickle.HIGHEST_PROTOCOL)

def randPair(s, e):
    return np.random.randint(s, e), np.random.randint(s, e)


# finds an array in the "depth" dimension of the grid
def findLoc(state, obj):
    #print(state)
    state = state.reshape((5,5,5))
    for i in range(0, 5):
        for j in range(0, 5):
            #print(state[i,j])
            if (state[i, j] == obj).all():
                return i, j


def get_grid_for_player(state, player):
    if np.array_equal(player, player_a):
        return state
    old_player_a_loc, old_player_b_loc, wall_loc, goal_loc, pit_loc = find_objects(state)
    state = np.zeros((5, 5, 5))
    state[old_player_a_loc] = player_b.copy()
    state[old_player_b_loc] = player_a.copy()
    # re-place pit
    state[pit_loc] = pit.copy()
    # re-place wall
    #state[wall_loc] = wall.copy()
    # re-place goal
    state[goal_loc] = goal.copy()
    return state


# Initialize stationary grid, all items are placed deterministically
'''
def init_grid():
    state = np.zeros((5, 5, 5))
    # place player a
    state[0, 1] = player_a.copy()
    # place player b
    state[0, 2] = player_b.copy()
    # place wall
    #state[2, 2] = wall.copy()
    # place pit
    state[1, 1] = pit.copy()
    # place goal
    state[3, 3] = goal.copy()
    return state
'''

# Initialize player in random location, but keep wall, goal and pit stationary
def init_grid_player():
    state = np.zeros((5, 5, 5))
    # place player a
    state[randPair(1, 3)] = player_a.copy()
    # place player b
    state[randPair(1, 3)] = player_b.copy()
    # place wall
    #state[2, 2] = wall.copy()
    # place pit
    state[4, 0] = pit.copy()
    # place goal
    state[0, 4] = goal.copy()

    player_a_loc, player_b_loc, wall_loc, goal_loc, pit_loc = find_objects(state)
    #print(find_objects(state))
    if not player_a_loc or not player_b_loc or not goal_loc or not pit_loc:
        #print('Invalid grid. Rebuilding..')
        return init_grid_player()
    #print(state.shape)
    return state.reshape((5,5,5))


# Initialize grid so that goal, pit, wall, player are all randomly placed
def init_grid_rand():
    state = np.zeros((5, 5, 5))
    # place player a
    state[randPair(0, 4)] = player_a.copy()
    # place player b
    state[randPair(0, 4)] = player_b.copy()
    # place wall
    #state[randPair(0, 4)] = wall.copy()
    # place pit
    state[4, 0] = pit.copy()
    # place goal
    state[0, 4] = goal.copy()

    player_a_loc, player_b_loc, wall_loc, goal_loc, pit_loc = find_objects(state)
    # If any of the "objects" are superimposed, just call the function again to re-place
    if not player_a_loc or not player_b_loc or not goal_loc or not pit_loc:
        # print('Invalid grid. Rebuilding..')
        return init_grid_rand()

    return state


def get_action(action):
    actions = np.array([ [1, 0],[-1, 0],[0, -1],[0, 1]])
    #[-1, 0], [1, 0], [0, -1], [0, 1]
    return actions[action]

def make_move(state, action_a, action_b):
    # need to locate player in grid
    # need to determine what object (if any) is in the new grid spot the player is moving to
    old_player_a_loc, old_player_b_loc, wall_loc, goal_loc, pit_loc = find_objects(state)
    state = np.zeros((5, 5, 5))

    actions = [[-1, 0], [1, 0], [0, -1], [0, 1]]
    # e.g. up => (player row - 1, player column + 0)
    new_player_a_loc = (old_player_a_loc[0] + actions[action_a][0], old_player_a_loc[1] + actions[action_a][1])
    if not in_bound(new_player_a_loc): #or new_player_a_loc == wall_loc:
        new_player_a_loc = old_player_a_loc

    new_player_b_loc = (old_player_b_loc[0] + actions[action_b][0], old_player_b_loc[1] + actions[action_b][1])
    player_b_moved = True
    if not in_bound(new_player_b_loc): #or new_player_b_loc == wall_loc:
        new_player_b_loc = old_player_b_loc
        player_b_moved = False

    if new_player_a_loc == new_player_b_loc and not player_b_moved:
        new_player_a_loc = old_player_a_loc
    state[new_player_a_loc][3] = 1

    if new_player_a_loc == new_player_b_loc:
        new_player_b_loc = old_player_b_loc
    state[new_player_b_loc][4] = 1

    new_player_a_loc = findLoc(state, player_a.copy())
    new_player_b_loc = findLoc(state, player_b.copy())
    if not (new_player_a_loc and new_player_b_loc):
        raise ValueError("Player A or B could not be found")
    # re-place pit
    state[pit_loc][1] = 1
    # re-place wall
    #state[wall_loc][2] = 1
    # re-place goal
    state[goal_loc][0] = 1

    return state


def in_bound(loc):
    if (np.array(loc) <= (4, 4)).all() and (np.array(loc) >= (0, 0)).all():
        return True
    return False


def get_loc(state, level):
    for i in range(0, 5):
        for j in range(0, 5):
            if state[i, j][level] == 1:
                return i, j


def get_reward(state):
    deterministic = True
    player_a_loc = get_loc(state, 3)
    player_b_loc = get_loc(state, 4)
    pit = get_loc(state, 1)
    goal = get_loc(state, 0)
   
    reward = 0
    player_a_terminal = False
    if deterministic :
        if player_a_loc == pit:
            reward -= 10
            
        elif player_a_loc == goal:
            reward += 10
        else:
            reward -= 1

        if player_b_loc == pit:
            reward -= 10
        elif player_b_loc == goal:
            reward += 10#
        else:
            reward -= 1
        # if player_a_loc == pit or player_b_loc == pit and not(player_a_loc == goal or player_b_loc == goal):
        #     reward += 20
        # elif player_a_loc == goal or player_b_loc == goal and not (player_a_loc == pit or player_b_loc == pit):
        #     reward +=20
        # elif player_a_loc == pit or player_b_loc == pit or player_a_loc == goal or player_b_loc == goal:
        #     reward += 20
        # else:
        #     reward -=2
    else:
        mu0, sigma0 = 20, 2
        mu1,sigma1 = 20,2#15,0.5
        s0 = np.random.normal(mu0, sigma0, 1)
        s1 = np.random.normal(mu1, sigma1, 1)

        #s[0] 
        # if player_a_loc == pit:
        #     reward += s0[0]
        # elif player_a_loc == goal:
        #     #mu, sigma = 0, 0.1 # mean and standard deviation
        #     reward += s1[0]
        # else:
        #     reward -= 1

        # if player_b_loc == pit:
        #     reward += s0[0]
        # elif player_b_loc == goal:
        #     reward += s1[0]
        # else:
        #     reward -= 1
        if player_a_loc == pit or player_b_loc == pit and not(player_a_loc == goal or player_b_loc == goal):
            reward += s0[0]
        elif player_a_loc == goal or player_b_loc == goal and not (player_a_loc == pit or player_b_loc == pit):
            reward +=s1[0]
        elif player_a_loc == pit or player_b_loc == pit or player_a_loc == goal or player_b_loc == goal:
            reward +=s1[0]/2+s0[0]/2
        else:
            reward -=.2

    return reward


def find_objects(state):
    player_a_loc = findLoc(state, player_a.copy())
    player_b_loc = findLoc(state, player_b.copy())
    wall_loc = findLoc(state, wall.copy())
    goal_loc = findLoc(state, goal.copy())
    pit_loc = findLoc(state, pit.copy())
    return player_a_loc, player_b_loc, wall_loc, goal_loc, pit_loc


def render(state):
    grid = np.zeros((5, 5), dtype=str)
    player_a_loc, player_b_loc, wall_loc, goal_loc, pit = find_objects(state)
    for i in range(0, 5):
        for j in range(0, 5):
            grid[i, j] = ' '

    if player_a_loc:
        grid[player_a_loc] = 'A'  # player A
    if player_b_loc:
        grid[player_b_loc] = 'B'  # player B
    if wall_loc:
        grid[wall_loc] = 'W'  # wall
    if goal_loc:
        grid[goal_loc] = '+'  # goal
    if pit:
        grid[pit] = '-'  # pit
    print(grid)


def is_done(state):
    player_a_loc, player_b_loc, wall_loc, goal_loc, pit_loc = find_objects(state)
    if not player_a_loc or not player_b_loc or not goal_loc or not pit_loc:
        return True
    else:
        return False
