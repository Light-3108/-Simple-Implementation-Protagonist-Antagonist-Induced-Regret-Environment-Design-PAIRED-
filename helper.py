import math
import random

import gym
import imageio

import numpy as np
import time 
import torch
import torch.nn as nn

import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
from IPython.display import clear_output
import matplotlib.pyplot as plt

from collections import deque
device = torch.device(0 if torch.cuda.is_available() else "cpu")


class ReplayBuffer:
    def __init__(self, data_names, buffer_size, mini_batch_size, device):
        self.data_keys = data_names
        self.data_dict = {}
        self.buffer_size = buffer_size
        self.mini_batch_size = mini_batch_size
        self.device = device

        self.reset()

    def reset(self):
        # Create a deque for each data type with set max length
        for name in self.data_keys:
            self.data_dict[name] = deque(maxlen=self.buffer_size)

    def buffer_full(self):
        return len(self.data_dict[self.data_keys[0]]) == self.buffer_size

    def data_log(self, data_name, data):
        # split tensor along batch into a list of individual datapoints
        data = data.cpu().split(1)
        # Extend the deque for data type, deque will handle popping old data to maintain buffer size
        self.data_dict[data_name].extend(data)

    def __iter__(self):
        batch_size = len(self.data_dict[self.data_keys[0]])
        batch_size = batch_size - batch_size % self.mini_batch_size

        ids = np.random.permutation(batch_size)
        ids = np.split(ids, batch_size // self.mini_batch_size)
        for i in range(len(ids)):
            batch_dict = {}
            for name in self.data_keys:
                c = [self.data_dict[name][j] for j in ids[i]]
                batch_dict[name] = torch.cat(c).to(self.device)
            batch_dict["batch_size"] = len(ids[i])
            yield batch_dict

    def __len__(self):
        return len(self.data_dict[self.data_keys[0]])
    # Procgen returns a dictionary as the state, this fuction converts the rbg images [0, 255] into a tensor [0, 1]

def state_to_tensor(obs, device): #[30,7,7,3]
    obs = torch.tensor(obs, dtype=torch.float32, device = device)
    obs = obs.permute(0, 3, 1, 2)  
    return obs  #[30,3,7,7]

# To test the agent we loop through all the training levels and an equivelant number of unseen levels
# Note this is not optimal as the training will wait untill this is done before continuing.
# With more training levels the time it takes to test will increase!
# Testing is usually done in a seperate process using the current saved checkpoint of the Policy parameters 
# (see IMPALA paper for a "full on" distributed method)
# def run_tests(dist_mode, env_name, num_levels, train_test="train"):
#     if train_test == "train":
#         start_level = 0
#     else:
#         start_level = num_levels
    
#     scores = []
#     for i in range(num_levels):
#         env = gym.make("procgen:procgen-" + env_name + "-v0", 
#                        start_level=start_level + i, num_levels=1, distribution_mode=dist_mode)
#         scores.append(test_agent(env))
        
#     return np.mean(scores)

# # Tests Policy once on the given environment
# def test_agent(env, log_states=False):
#     start_state = env.reset()
#     state = state_to_tensor(start_state, device)
    
#     if log_states:
#         states_logger = [tensor_to_unit8(state)]
    
#     done = False
#     total_reward = 0
#     with torch.no_grad():
#         while not done:
#             dist, _ = rl_model(state)  # Forward pass of actor-critic model
#             action = dist.sample().item()

#             next_state, reward, done, _ = env.step(action)
#             total_reward += reward
#             state = state_to_tensor(next_state, device)
#             if log_states:
#                 states_logger.append(tensor_to_unit8(state))
                
#     if log_states:
#         return total_reward, states_logger
#     else:
#         return total_reward
    
def ppo_loss(new_dist, actions, old_log_probs, advantages, clip_param):

    new_log_probs = new_dist.log_prob(actions)
    ratio = (new_log_probs - old_log_probs).exp()
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantages
    actor_loss = torch.min(surr1, surr2)
    return actor_loss.mean()

def clipped_critic_loss(new_value, old_value, returns, clip_param):

    vf_loss1 = (new_value - returns).pow(2.)
    vpredclipped = old_value + torch.clamp(new_value - old_value, -clip_param, clip_param)
    vf_loss2 = (vpredclipped - returns).pow(2.)
    critic_loss = torch.max(vf_loss1, vf_loss2)
    return critic_loss.mean()

def compute_gae(next_value, rewards, masks, values, gamma=0.999, tau=0.95):
    # Similar to calculating the returns we can start at the end of the sequence and go backwards
    gae = 0
    returns = deque()
    gae_logger = deque()

    for step in reversed(range(len(rewards))):
        # Calculate the current delta value
        delta = rewards[step] + gamma * next_value * masks[step] - values[step]
        
        # The GAE is the decaying sum of these delta values
        gae = delta + gamma * tau * masks[step] * gae
        
        # Get the new next value
        next_value = values[step]
        
        # If we add the value back to the GAE we get a TD approximation for the returns
        # which we can use to train the Value function
        returns.appendleft(gae + values[step])
        gae_logger.appendleft(gae)

    return returns, gae_logger


