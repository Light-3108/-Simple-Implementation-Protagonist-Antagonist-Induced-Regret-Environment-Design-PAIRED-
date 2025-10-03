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
    ########### Policy Gradient update for actor with clipping - PPO #############
    
    # 1. Find the new probability 
    # Work out the probability (log probability) that the agent will NOW take
    # the action it took during the rollout
    # We assume there has been some optimisation steps between when the action was taken and now so the
    # probability has probably changed
    new_log_probs = new_dist.log_prob(actions)
    
    # 2. Find the ratio of new to old - r_t(theta)
    # Calculate the ratio of new/old action probability (remember we have log probabilities here)
    # log(new_prob) - log(old_prob) = log(new_prob/old_prob)
    # exp(log(new_prob/old_prob)) = new_prob/old_prob
    # We use the ratio of new/old action probabilities (not just the log probability of the action like in
    # vanilla policy gradients) so that if there is a large difference between the probabilities then we can
    # take a larger/smaller update step
    # EG: If we want to decrease the probability of taking an action but the new action probability
    # is now higher than it was before we can take a larger update step to correct this
    ratio = (new_log_probs - old_log_probs).exp()

    # 3. Calculate the ratio * advantage - the first term in the MIN statement
    # We want to MAXIMISE the (Advantage * Ratio)
    # If the advantage is positive this corresponds to INCREASING the probability of taking that action
    # If the advantage is negative this corresponds to DECREASING the probability of taking that action
    surr1 = ratio * advantages

    # 4. Calculate the (clipped ratio) * advantage - the second term in the MIN statement
    # PPO goes a bit further, if we simply update update using the Advantage * Ratio we will sometimes
    # get very large or very small policy updates when we don't want them
    #
    # EG1: If we want to increase the probability of taking an action but the new action probability
    # is now higher than it was before we will take a larger step, however if the action probability is
    # already higher we don't need to keep increasing it (large output values can create instabilities).
    #
    # EG2: You can also consider the opposite case where we want to decrease the action probability
    # but the probability has already decreased, in this case we will take a smaller step than before,
    # which is also not desirable as it will slow down the "removal" (decreasing the probability)
    # of "bad" actions from our policy.
    surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantages
    
    # 5. Take the minimum of the two "surrogate" losses
    # PPO therefore clips the upper bound of the ratio when the advantage is positive
    # and clips the lower bound of the ratio when the advantage is negative so our steps are not too large
    # or too small when necessary, it does this by using a neat trick of simply taking the MIN of two "surrogate"
    # losses which chooses which loss to use!
    actor_loss = torch.min(surr1, surr2)
    
    # 6. Return the Expectation over the batch
    return actor_loss.mean()

def clipped_critic_loss(new_value, old_value, returns, clip_param):
    ########### Value Function update for critic with clipping #############
    
    # To help stabalise the training of the value function we can do a similar thing as the clipped objective
    # for PPO - Note: this is NOT nessisary but does help!
        
    # 1. MSE/L2 loss on the current value and the returns
    vf_loss1 = (new_value - returns).pow(2.)
    
    # 2. MSE/L2 loss on the clipped value and the returns
    # Here we create an "approximation" of the new value (aka the current value) by finding the difference
    # between the "new" and "old" value and adding a clipped amount back to the old value
    vpredclipped = old_value + torch.clamp(new_value - old_value, -clip_param, clip_param)
    # Note that we ONLY backprop through the new value
    vf_loss2 = (vpredclipped - returns).pow(2.)
    
    # 3. Take the MAX between the two losses
    # This trick has the effect of only updating the current value DIRECTLY if is it WORSE (higher error)
    # than the old value. 
    # If the old value was worse then the "approximation" will be worse and we update
    # the new value only a little bit!
    critic_loss = torch.max(vf_loss1, vf_loss2)
    
    # 4. Return the Expectation over the batch
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


