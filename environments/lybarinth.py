# NO LSTM
# Simple env 
# fully observable
# (10,10,3) -> wall at sides so -> (8,8,3)
from __future__ import annotations
from collections import deque
from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Wall
from minigrid.manual_control import ManualControl
from minigrid.minigrid_env import MiniGridEnv
from minigrid.wrappers import FullyObsWrapper

import numpy as np
import gymnasium as gym
from gymnasium.vector import SyncVectorEnv, AsyncVectorEnv

class SimpleEnv(MiniGridEnv):
    def __init__(
        self,
        size=10,
        max_steps: int | None = None,
        **kwargs,
    ):
        self.agent_start_pos = (1,1)  # this will be overwritten by NN
        self.agent_start_dir = 0 # this too. 

        mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = 4 * (size-2)**2

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            # Set this to True for maximum speed
            see_through_walls=True,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return "grand mission"
    
        
    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # # Random agent position (avoid goal position)

        self.put_obj(Wall(),1,7)
        self.put_obj(Wall(),2,7)
        self.put_obj(Wall(),6,7)
        for i in range(2,8):
            self.put_obj(Wall(),i,2)
        
        for i in range(2,8):
            self.put_obj(Wall(),7,i)
        
        for i in range(2,6):
            self.put_obj(Wall(),2,i)
        
        for i in range(4,8):
            self.put_obj(Wall(),4,i)

        self.put_obj(Wall(),3,5)
        self.put_obj(Wall(),4,8)
        self.agent_pos = (1, 8)
        self.agent_dir = np.random.randint(0, 4)
        self.put_obj(Goal(), 3,4)

        



def make_env():

    def _init():
        env = SimpleEnv(render_mode = "rgb_array")
        env = FullyObsWrapper(env)
        return env
    return _init

    
def main():

    env = SimpleEnv(render_mode = "human")

    # env_fns = [make_env() for _ in range(30)]
    # envs = SyncVectorEnv(env_fns)

    # observations, infos = envs.reset()

    # for _ in range(3):

    #     actions = envs.action_space.sample()
    #     observations, rewards, terminations, truncations, infos = envs.step(actions)
    #     print(actions)
    # envs.close()

    manual_control = ManualControl(env, seed=42)
    manual_control.start()

    # obs, info = env.reset()

    # print(obs['image'])  
    # while not done:
    #     action = np.random.randint(0,3)
    #     obs, reward, terminated, truncated, info = env.step(action)
    #     done = terminated or truncated
    #     print(action)
    #     env.render()
    # env.close()
  
if __name__ == "__main__":
    main()


