# from gridworld import Goldmine as GAME_ENV
# from gridworld import state_size,get_grid_for_player,v_state,number_of_eval_games

import numpy as np
import gymnasium as gym
import minigrid
from minigrid.wrappers import FlatObsWrapper

'''
Envs:

env = MiniGrid-Empty-6x6-v0 , action = [0,1,2],  number_of_eval_games = 96

Complexity: Easy

env = MiniGrid-LavaCrossingS9N3-v0 , aciton = [0,1,2],  number_of_eval_games = 176
Desc: The agent has to reach the green goal square on the other corner of the room while avoiding 
rivers of deadly lava which terminate the episode in failure. 
Each lava stream runs across the room either horizontally or vertically, 
and has a single crossing point which can be safely used; Luckily, a path to the goal is guaranteed to exist. 
This environment is useful for studying safety and safe exploration.


Complexity: Hard


env = MiniGrid-LavaGapS5-v0,  action = [0,1,2], number_of_eval_games = 24
Desc: The agent has to reach the green goal square at the opposite corner of the room, 
and must pass through a narrow gap in a vertical strip of deadly lava. 
Touching the lava terminate the episode with a zero reward. 
This environment is useful for studying safety and safe exploration.

Complexity: Medium
'''

env = gym.make("MiniGrid-LavaGapS5-v0")
number_of_eval_games = 25
observation, info = env.reset(seed=42)
GAME_ENV = FlatObsWrapper(env)
obs, _ = GAME_ENV.reset()
ACTION_SPACE_LIST = [0,1,2]
ACTION_SPACE= len(ACTION_SPACE_LIST)

state_size = obs.shape[0]
