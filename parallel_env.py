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
import matplotlib.pyplot as plt
import copy
import torch
import numpy as np
import gymnasium as gym
from gymnasium.vector import SyncVectorEnv, AsyncVectorEnv
from adversary import Adversary
import seed_counter
class SimpleEnv(MiniGridEnv):
    def __init__(
        self,
        adversary_agent, 
        size=10,
        log_probs_ad=None, values_ad=None, states_ad=None,
        actions_ad=None, rewards_ad=None, masks_ad=None,
        time_ad = None, z_ad = None,
        max_steps: int | None = None,
        **kwargs,
    ):
        self.adversary_agent = adversary_agent
        self.log_probs_ad = log_probs_ad if log_probs_ad is not None else deque()
        self.values_ad = values_ad if values_ad is not None else deque()
        self.states_ad = states_ad if states_ad is not None else deque()
        self.actions_ad = actions_ad if actions_ad is not None else deque()
        self.rewards_ad = rewards_ad if rewards_ad is not None else deque()
        self.masks_ad = masks_ad if masks_ad is not None else deque()
        self.time_ad = time_ad if time_ad is not None else deque()
        self.z_ad = z_ad if z_ad is not None else deque()


        self.agent_start_pos = (1,1)  # this will be overwritten by NN
        self.agent_start_dir = 0 # this too. 

        self.initial_agent_pos = None
        self.initial_agent_dir = None

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


    # def seed(self):
    #     """Always deterministic RNG based on base_seed."""
    #     self.np_random = np.random.default_rng(self.base_seed)
    #     self.torch_gen = torch.Generator().manual_seed(self.base_seed)

    def reset(self, *, seed=None, options=None):

        # Clear history containers
        self.log_probs_ad.clear()
        self.values_ad.clear()
        self.states_ad.clear()
        self.actions_ad.clear()
        self.rewards_ad.clear()
        self.masks_ad.clear()
        self.time_ad.clear()
        self.z_ad.clear()
        
        if options is not None and "keep_world" in options:
            # while True:
            #     x = np.random.randint(1, self.width - 1)
            #     y = np.random.randint(1, self.height - 1)
            #     if self.grid.get(x,y) is None:
            #         break
            self.step_count = 0
            self.agent_pos = self.initial_agent_pos
            self.agent_dir = self.initial_agent_dir
        else:
            super().reset(seed=seed, options=options)

        obs = self.gen_obs()
        info = {}
        return obs, info
    
    @staticmethod
    def _gen_mission():
        return "grand mission"
    
    def check_valid_positions(self,x,y, exclude_pos=None, exclude_pos2=None):
        
        if self.grid.get(x, y) is None and (exclude_pos is None or (x, y) != exclude_pos) and (exclude_pos2 is None or (x, y) != exclude_pos2):
            return (x, y)
        else:
            return (-1,-1)
        
    def _gen_grid(self, width, height):
        # Create an empty grid

        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)
        # x = np.random.randint(1,8)
        # y = np.random.randint(1,8)
        # self.agent_pos = (x, y)
        # self.agent_dir = np.random.randint(1,4)
        # x1 = np.random.randint(1,8)
        # y1 = np.random.randint(1,8)
        # self.put_obj(Goal(), x1, y1)
        self.agent_pos = None # gen_obs() crash hune raixa, so paile nai initialize gareko 
        self.agent_dir = None      # paxi overwrite hunxa
        self.goal_pos = None

        obs = self.grid.encode()  # full obs ko lagi
        # Use a single latent z per episode (environment generation) so the maze structure is coherent
        with torch.no_grad():
            z_tensor = torch.randn(1, 20)  # (1,20) fixed for this episode
            for t in range(20):
                obs_tensor = torch.from_numpy(obs).float().unsqueeze(0)  # (1,15,15,3)
                obs_tensor = obs_tensor.permute(0, 3, 1, 2)  # (1,3,15,15)
                t_tensor = torch.tensor([[t]], dtype=torch.float32)  # (1,1)

                dist, value = self.adversary_agent(obs_tensor, t_tensor, z_tensor)
                obs_tensor_for_store = obs_tensor.permute(0,2,3,1)  # back to (1,15,15,3)
                action = dist.sample().squeeze(-1)
                log_prob = dist.log_prob(action)

                # Store rollout data (no need for .detach() inside no_grad, kept implicit)
                self.states_ad.append(obs_tensor_for_store)
                self.actions_ad.append(action)
                self.log_probs_ad.append(log_prob)
                self.values_ad.append(value)
                self.rewards_ad.append(0.0)
                self.masks_ad.append(1.0 if t < 19 else 0.0)
                self.time_ad.append(t_tensor)
                self.z_ad.append(z_tensor)

                # Decode action -> place objects
                a = action.item() + 1
                x = (a - 1) % 8 + 1
                y = (a - 1) // 8 + 1
                if t == 0:
                    self.agent_pos = (x, y)
                    self.agent_dir = np.random.randint(0,4)
                    self.initial_agent_pos = (x,y)
                    self.initial_agent_dir = self.agent_dir
                elif t == 1:
                    x, y = self.check_valid_positions(x, y, self.agent_pos)
                    if x != -1:
                        self.put_obj(Goal(), x, y)
                        self.goal_pos = (x, y)
                    else:
                        # place the goal randomly if clash
                        while True:
                            x = np.random.randint(1, self.width - 1)
                            y = np.random.randint(1, self.height - 1)
                            if (x, y) != self.agent_pos:
                                break
                        self.put_obj(Goal(), x, y)
                        self.goal_pos = (x, y)
                else:
                    x, y = self.check_valid_positions(x, y, self.agent_pos, self.goal_pos)
                    if x != -1:
                        self.put_obj(Wall(), x, y)
                obs = self.grid.encode()
        self.mission = "grand mission"


def make_env(agent = None):
    def _init():
        env = SimpleEnv(adversary_agent=agent, render_mode="rgb_array")
        env = FullyObsWrapper(env)
        return env
    return _init




def main():

    adversary = Adversary()
    num_envs = 30
    env_fns = [make_env(agent=adversary) for i in range(num_envs)]
    env = SyncVectorEnv(env_fns)

    # env1 = env.envs[0]
    obs, info = env.reset()
    env1 = env.envs[0]
    grid_img = env1.render()
    plt.imshow(grid_img)
    plt.title(f"Grid at {1} frames")
    plt.savefig(f"grid_at_{1}.png")
    plt.close()

    obs, info = env.reset(options = {"keep_world": True})
    env1 = env.envs[0]
    grid_img = env1.render()
    plt.imshow(grid_img)
    plt.title(f"Grid at {2} frames")
    plt.savefig(f"grid_at_{2}.png")
    plt.close()
    obs, info = env.reset() 

    # for env in env.envs:
    #     env.env.step_count = 18
    #     print(env.env.step_count)   # go to simple env from full obs wrapper i.e env.env

if __name__ == "__main__":
    main()


