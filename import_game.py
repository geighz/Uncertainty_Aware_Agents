#from gridworld import Goldmine as GAME_ENV
from gridworld import state_size,get_grid_for_player,v_state,number_of_eval_games

import numpy as np
import gymnasium as gym
import minigrid
from minigrid.wrappers import FlatObsWrapper

env = gym.make("MiniGrid-Empty-6x6-v0")
observation, info = env.reset(seed=42)
GAME_ENV = FlatObsWrapper(env)
obs, _ = GAME_ENV.reset()
ACTION_SPACE_LIST = [0,1,2]
ACTION_SPACE= len(ACTION_SPACE_LIST)

state_size = obs.shape[0]
