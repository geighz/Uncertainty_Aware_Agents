import numpy as np


def randPair(s, e):
    return np.random.randint(s, e), np.random.randint(s, e)


# finds an array in the "depth" dimension of the grid
def findLoc(state, obj):
    for i in range(0, 4):
        for j in range(0, 4):
            if (state[i, j] == obj).all():
                return i, j


# Initialize stationary grid, all items are placed deterministically
def initGrid():
    state = np.zeros((4, 4, 5))
    # place player a
    state[0, 1] = np.array([0, 0, 0, 1, 0])
    # place player b
    state[0, 2] = np.array([0, 0, 0, 0, 1])
    # place wall
    state[2, 2] = np.array([0, 0, 1, 0, 0])
    # place pit
    state[1, 1] = np.array([0, 1, 0, 0, 0])
    # place goal
    state[3, 3] = np.array([1, 0, 0, 0, 0])
    return state


# Initialize player in random location, but keep wall, goal and pit stationary
def init_grid_player():
    state = np.zeros((4, 4, 5))
    # place player
    state[randPair(0, 4)] = np.array([0, 0, 0, 1, 0])
    # place player
    state[randPair(0, 4)] = np.array([0, 0, 0, 0, 1])
    # place wall
    state[2, 2] = np.array([0, 0, 1, 0, 0])
    # place pit
    state[1, 1] = np.array([0, 1, 0, 0, 0])
    # place goal
    state[3, 3] = np.array([1, 0, 0, 0, 0])

    player_a_loc, player_b_loc, wall, goal, pit = find_objects(state)
    if not player_a_loc or not player_b_loc or not wall or not goal or not pit:
        # print('Invalid grid. Rebuilding..')
        return init_grid_player()

    return state


# Initialize grid so that goal, pit, wall, player are all randomly placed
def init_grid_rand():
    state = np.zeros((4, 4, 5))
    # place player
    state[randPair(0, 4)] = np.array([0, 0, 0, 1, 0])
    # place player
    state[randPair(0, 4)] = np.array([0, 0, 0, 0, 1])
    # place wall
    state[randPair(0, 4)] = np.array([0, 0, 1, 0, 0])
    # place pit
    state[randPair(0, 4)] = np.array([0, 1, 0, 0, 0])
    # place goal
    state[randPair(0, 4)] = np.array([1, 0, 0, 0, 0])

    player_a_loc, player_b_loc, wall, goal, pit = find_objects(state)
    # If any of the "objects" are superimposed, just call the function again to re-place
    if not player_a_loc or not player_b_loc or not wall or not goal or not pit:
        # print('Invalid grid. Rebuilding..')
        return init_grid_rand()

    return state


def make_move(state, action_a, action_b):
    # need to locate player in grid
    # need to determine what object (if any) is in the new grid spot the player is moving to
    old_player_a_loc, old_player_b_loc, wall, goal, pit = find_objects(state)
    state = np.zeros((4, 4, 5))

    actions = [[-1, 0], [1, 0], [0, -1], [0, 1]]
    # e.g. up => (player row - 1, player column + 0)
    new_player_a_loc = (old_player_a_loc[0] + actions[action_a][0], old_player_a_loc[1] + actions[action_a][1])
    if not in_bound(new_player_a_loc) or new_player_a_loc == wall:
        new_player_a_loc = old_player_a_loc

    new_player_b_loc = (old_player_b_loc[0] + actions[action_b][0], old_player_b_loc[1] + actions[action_b][1])
    player_b_moved = True
    if not in_bound(new_player_b_loc) or new_player_b_loc == wall:
        new_player_b_loc = old_player_b_loc
        player_b_moved = False

    if new_player_a_loc == new_player_b_loc and not player_b_moved:
        new_player_a_loc = old_player_a_loc
    state[new_player_a_loc][3] = 1

    if new_player_a_loc == new_player_b_loc:
        new_player_b_loc = old_player_b_loc
    state[new_player_b_loc][4] = 1

    new_player_a_loc = findLoc(state, np.array([0, 0, 0, 1, 0]))
    new_player_b_loc = findLoc(state, np.array([0, 0, 0, 1, 0]))
    if not (new_player_a_loc and new_player_b_loc):
        raise ValueError("Player A or B could not be found")
    # re-place pit
    state[pit][1] = 1
    # re-place wall
    state[wall][2] = 1
    # re-place goal
    state[goal][0] = 1

    return state


def in_bound(loc):
    if (np.array(loc) <= (3, 3)).all() and (np.array(loc) >= (0, 0)).all():
        return True
    return False


def get_loc(state, level):
    for i in range(0, 4):
        for j in range(0, 4):
            if state[i, j][level] == 1:
                return i, j


def get_reward(state):
    player_a_loc = get_loc(state, 3)
    player_b_loc = get_loc(state, 4)
    pit = get_loc(state, 1)
    goal = get_loc(state, 0)
    reward = 0
    if player_a_loc == pit:
        reward -= 10
    elif player_a_loc == goal:
        reward += 10
    else:
        reward -= 1

    if player_b_loc == pit:
        reward -= 10
    elif player_b_loc == goal:
        reward += 10
    else:
        reward -= 1

    return reward


def find_objects(state):
    player_a_loc = findLoc(state, np.array([0, 0, 0, 1, 0]))
    player_b_loc = findLoc(state, np.array([0, 0, 0, 0, 1]))
    wall = findLoc(state, np.array([0, 0, 1, 0, 0]))
    goal = findLoc(state, np.array([1, 0, 0, 0, 0]))
    pit = findLoc(state, np.array([0, 1, 0, 0, 0]))
    return player_a_loc, player_b_loc, wall, goal, pit


def disp_grid(state):
    grid = np.zeros((4, 4), dtype=str)
    player_a_loc, player_b_loc, wall, goal, pit = find_objects(state)
    for i in range(0, 4):
        for j in range(0, 4):
            grid[i, j] = ' '

    if player_a_loc:
        grid[player_a_loc] = 'A'  # player A
    if player_b_loc:
        grid[player_b_loc] = 'B'  # player B
    if wall:
        grid[wall] = 'W'  # wall
    if goal:
        grid[goal] = '+'  # goal
    if pit:
        grid[pit] = '-'  # pit

    return grid
