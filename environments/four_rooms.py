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

        for i in range(2,5):
            self.put_obj(Wall(),i,5)
        for i in range(5,7):
            self.put_obj(Wall(),i,5)
        for i in range(7,9):
            self.put_obj(Wall(),i,5)
        
        for i in range(1,2):
            self.put_obj(Wall(),5,i)
        for i in range(3,7):
            self.put_obj(Wall(),5,i)
        for i in range(7,8):
            self.put_obj(Wall(),5,i)
        

        # while True:
        #     x = np.random.randint(1, 8)
        #     y = np.random.randint(1, 8)
        #     x1 = np.random.randint(1, 8)
        #     y1 = np.random.randint(1, 8)
            
        #     # Ensure agent and goal are not at the same position
        #     if (x, y) != (x1, y1) and self.grid.get(x, y) is None and self.grid.get(x1, y1) is None:
        #         break
               
        # self.agent_pos = (x, y)
        # self.agent_dir = np.random.randint(0, 4)  # 0-3 for directions
        # self.put_obj(Goal(), x1, y1)

        self.agent_pos = (7, 8)
        self.agent_dir = np.random.randint(0, 4)  # 0-3 for directions
        self.put_obj(Goal(), 2, 2)

        



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


