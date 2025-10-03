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
        adversary_agent,   # Gradient flow roknu parla ata ni
        base_seed: int,
        size=10,
        log_probs_ad=None, values_ad=None, states_ad=None,
        actions_ad=None, rewards_ad=None, masks_ad=None,
        time_ad = None, z_ad = None,
        max_steps: int | None = None,
        **kwargs,
    ):
        self.base_seed = base_seed 
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

        self._seed = None
        self.np_random = None
        self.torch_gen = None

    # def seed(self):
    #     """Always deterministic RNG based on base_seed."""
    #     self.np_random = np.random.default_rng(self.base_seed)
    #     self.torch_gen = torch.Generator().manual_seed(self.base_seed)

    def reset(self, *, seed=None, options=None):


        #manual control garnu parye
        # do varied_seed = self.base_seed when options is None
        varied_seed = self.base_seed + seed_counter.seed_count
        self.np_random = np.random.default_rng(varied_seed)
        self.torch_gen = torch.Generator().manual_seed(varied_seed)
        
        # Clear history containers
        self.log_probs_ad.clear()
        self.values_ad.clear()
        self.states_ad.clear()
        self.actions_ad.clear()
        self.rewards_ad.clear()
        self.masks_ad.clear()
        self.time_ad.clear()
        self.z_ad.clear()
        
        return super().reset(seed=seed, options=options)
    
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
        with torch.no_grad():
            for t in range(20):

                obs_tensor = torch.from_numpy(obs).float().unsqueeze(0)  # (1,15,15,3)
                obs_tensor = obs_tensor.permute(0, 3, 1, 2)  # (1,3,15,15)
                t_tensor = torch.tensor([[t]], dtype=torch.float32)  # (1,1)
                z_tensor = torch.randn(1, 10, generator=self.torch_gen)  # (1,50) # every episode ma different maze aayos vanera ho

                dist, value = self.adversary_agent(obs_tensor, t_tensor, z_tensor)
                obs_tensor = obs_tensor.permute(0,2,3,1)
                probs = dist.probs
                action = torch.multinomial(probs, num_samples=1, generator=self.torch_gen).squeeze(-1)
                log_prob = dist.log_prob(action)
                # prob = F.softmax(policy_logits, dim=-1)  # (1,169)
                # action = torch.multinomial(prob, num_samples=1).item() #(1,1)

                self.states_ad.append(obs_tensor)
                self.actions_ad.append(action)  
                self.log_probs_ad.append(log_prob)
                self.values_ad.append(value)
                self.rewards_ad.append(0.0) 
                if (t < 19):
                    self.masks_ad.append(1.0)
                else:
                    self.masks_ad.append(0.0)
                self.time_ad.append(t_tensor)
                self.z_ad.append(z_tensor)

                action = action.item()
                action += 1
                x = (action - 1) % 8 + 1
                y = (action - 1) // 8 + 1
                if t == 0:
                    self.agent_pos = (x, y)
                    self.agent_dir = 0
                elif t == 1:
                    x,y = self.check_valid_positions(x,y,self.agent_pos)
                    if x != -1:
                        self.put_obj(Goal(), x, y)
                        self.goal_pos = (x, y)
                    else:
                        #place the goal randomly
                        while True:
                            x = self.np_random.integers(1, self.width - 1)
                            y = self.np_random.integers(1, self.height - 1)
                            if (x,y) != self.agent_pos:
                                break
                        self.put_obj(Goal(), x, y)
                        self.goal_pos = (x, y)
                else:  
                    x,y = self.check_valid_positions(x,y,self.agent_pos,self.goal_pos)
                    if x != -1:
                        self.put_obj(Wall(), x, y)
                obs = self.grid.encode()  # update obs after each placement
        self.mission = "grand mission"


def make_env(rank, global_seed=42, agent = None):
    def _init():
        base_seed = global_seed + rank
        env = SimpleEnv(adversary_agent=agent, base_seed=base_seed, render_mode="rgb_array")
        env = FullyObsWrapper(env)
        return env
    return _init




def main():



    # env = make_env(0, global_seed=42)()
    adversary = Adversary()
    adversary.load_state_dict(torch.load("Adversary_1200.pth"))
    adversary.eval()
    num_envs = 30
    env_fns = [make_env(rank=i, global_seed=42, agent=adversary) for i in range(num_envs)]
    env = SyncVectorEnv(env_fns)

    env1 = env.envs[0]
    manual_control = ManualControl(env1)
    manual_control.start()
    # obs, infos = env.reset(options={"should_regenerate": False})
    # obs1 = obs['image'][0]
    # obs, infos = env.reset(options={"should_regenerate": False})
    # obs2 = obs['image'][0]

    # obs, infos = env.reset(options=20)
    # obs1 = obs['image'][0]
    # obs2 = obs['image'][1]
    # # # obs, infos = env.reset(options=20)
    # # # obs2 = obs['image'][0]
    # assert np.array_equal(obs1, obs2)
    # obs, infos = env.reset(options=21)
    # obs3 = obs['image'][0]
    # obs, infos = env.reset(options=21)
    # obs4 = obs['image'][0]

    # assert np.array_equal(obs1, obs2)
    # assert np.array_equal(obs3, obs4)
    # assert not np.array_equal(obs1, obs3)

if __name__ == "__main__":
    main()



