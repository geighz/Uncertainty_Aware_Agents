import numpy as np

player_a = np.array([0, 0, 0, 1, 0])
player_b = np.array([0, 0, 0, 0, 1])
wall = np.array([0, 0, 1, 0, 0])
pit = np.array([0, 1, 0, 0, 0])
goal = np.array([1, 0, 0, 0, 0])

def randPair(s, e):
    return np.random.randint(s, e), np.random.randint(s, e)


# finds an array in the "depth" dimension of the grid
def findLoc(state, obj):
    for i in range(0, 4):
        for j in range(0, 4):
            if (state[i, j] == obj).all():
                return i, j


def get_grid_for_player(state, player):
    if np.array_equal(player, player_a):
        return state
    old_player_a_loc, old_player_b_loc, wall_loc, goal_loc, pit_loc = find_objects(state)
    state = np.zeros((4, 4, 5))
    state[old_player_a_loc] = player_b.copy()
    state[old_player_b_loc] = player_a.copy()
    # re-place pit
    state[pit_loc] = pit.copy()
    # re-place wall
    state[wall_loc] = wall.copy()
    # re-place goal
    state[goal_loc] = goal.copy()
    return state


# Initialize stationary grid, all items are placed deterministically
def init_grid():
    state = np.zeros((4, 4, 5))
    # place player a
    state[0, 1] = player_a.copy()
    # place player b
    state[0, 2] = player_b.copy()
    # place wall
    state[2, 2] = wall.copy()
    # place pit
    state[1, 1] = pit.copy()
    # place goal
    state[3, 3] = goal.copy()
    return state


# Initialize player in random location, but keep wall, goal and pit stationary
def init_grid_player():
    state = np.zeros((4, 4, 5))
    # place player a
    state[randPair(0, 4)] = player_a.copy()
    # place player b
    state[randPair(0, 4)] = player_b.copy()
    # place wall
    state[2, 2] = wall.copy()
    # place pit
    state[1, 1] = pit.copy()
    # place goal
    state[3, 3] = goal.copy()

    player_a_loc, player_b_loc, wall_loc, goal_loc, pit_loc = find_objects(state)
    if not player_a_loc or not player_b_loc or not wall_loc or not goal_loc or not pit_loc:
        # print('Invalid grid. Rebuilding..')
        return init_grid_player()

    return state


# Initialize grid so that goal, pit, wall, player are all randomly placed
def init_grid_rand():
    state = np.zeros((4, 4, 5))
    # place player a
    state[randPair(0, 4)] = player_a.copy()
    # place player b
    state[randPair(0, 4)] = player_b.copy()
    # place wall
    state[randPair(0, 4)] = wall.copy()
    # place pit
    state[randPair(0, 4)] = pit.copy()
    # place goal
    state[randPair(0, 4)] = goal.copy()

    player_a_loc, player_b_loc, wall_loc, goal_loc, pit_loc = find_objects(state)
    # If any of the "objects" are superimposed, just call the function again to re-place
    if not player_a_loc or not player_b_loc or not wall_loc or not goal_loc or not pit_loc:
        # print('Invalid grid. Rebuilding..')
        return init_grid_rand()

    return state


def make_move(state, action_a, action_b):
    # need to locate player in grid
    # need to determine what object (if any) is in the new grid spot the player is moving to
    old_player_a_loc, old_player_b_loc, wall_loc, goal_loc, pit_loc = find_objects(state)
    state = np.zeros((4, 4, 5))

    actions = [[-1, 0], [1, 0], [0, -1], [0, 1]]
    # e.g. up => (player row - 1, player column + 0)
    new_player_a_loc = (old_player_a_loc[0] + actions[action_a][0], old_player_a_loc[1] + actions[action_a][1])
    if not in_bound(new_player_a_loc) or new_player_a_loc == wall_loc:
        new_player_a_loc = old_player_a_loc

    new_player_b_loc = (old_player_b_loc[0] + actions[action_b][0], old_player_b_loc[1] + actions[action_b][1])
    player_b_moved = True
    if not in_bound(new_player_b_loc) or new_player_b_loc == wall_loc:
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
    state[wall_loc][2] = 1
    # re-place goal
    state[goal_loc][0] = 1

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
    player_a_loc = findLoc(state, player_a.copy())
    player_b_loc = findLoc(state, player_b.copy())
    wall_loc = findLoc(state, wall.copy())
    goal_loc = findLoc(state, goal.copy())
    pit_loc = findLoc(state, pit.copy())
    return player_a_loc, player_b_loc, wall_loc, goal_loc, pit_loc


def disp_grid(state):
    grid = np.zeros((4, 4), dtype=str)
    player_a_loc, player_b_loc, wall_loc, goal_loc, pit = find_objects(state)
    for i in range(0, 4):
        for j in range(0, 4):
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

    return grid
