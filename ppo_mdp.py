from pandas import options
from helper import *
import math
import random
import seed_counter
import gymnasium as gym

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
from protagonist_mdp import Protagonist
from parallel_env import make_env
from adversary import Adversary
from antagonist import Antagonist
from gymnasium.vector import SyncVectorEnv, AutoresetMode
num_envs = 30
device = torch.device(0 if torch.cuda.is_available() else "cpu")
print("Using device:", device)
lr = 0.0001
# How many minibatchs (therefore optimization steps) we want per epoch 
num_mini_batch = 1
num_mini_batch_ad = 1
# Total number of steps during the rollout phase 
num_steps = 256 
num_steps_ad = 20
# Number of Epochs for training
ppo_epochs = 5

# PPO parameters
gamma = 0.995
tau = 0.95
clip_param = 0.2



def ppo_update(data_buffer, ppo_epochs, clip_param):
    for _ in range(ppo_epochs):
        for data_batch in data_buffer:

            new_dist, new_value = rl_model(data_batch["states"], data_batch["directions"])
            entropy = new_dist.entropy().mean()

            actor_loss = ppo_loss(new_dist, data_batch["actions"], data_batch["log_probs"], data_batch["advantages"],
                                  clip_param)

            critic_loss = clipped_critic_loss(new_value, data_batch["values"], data_batch["returns"], clip_param)
            agent_loss = critic_loss - actor_loss

            optimizer.zero_grad()
            agent_loss.backward()
            nn.utils.clip_grad_norm_(rl_model.parameters(), 40)
            optimizer.step()
        if _ == ppo_epochs - 1:  # Only log last epoch
            print(f"  Loss_protagonist - Actor: {actor_loss.item():.4f}, Critic: {critic_loss.item():.4f}, Entropy: {entropy.item():.4f}")


def ppo_update_an(data_buffer, ppo_epochs, clip_param):
    for _ in range(ppo_epochs):
        for data_batch in data_buffer:

            new_dist, new_value = an_model(data_batch["states"], data_batch["directions"])
            entropy = new_dist.entropy().mean()

            actor_loss = ppo_loss(new_dist, data_batch["actions"], data_batch["log_probs"], data_batch["advantages"],
                                  clip_param)

            critic_loss = clipped_critic_loss(new_value, data_batch["values"], data_batch["returns"], clip_param)
            agent_loss = critic_loss - actor_loss

            an_optimizer.zero_grad()
            agent_loss.backward()
            nn.utils.clip_grad_norm_(an_model.parameters(), 40)
            an_optimizer.step()
        if _ == ppo_epochs - 1:  # Only log last epoch
            print(f"  Loss_antagonist - Actor: {actor_loss.item():.4f}, Critic: {critic_loss.item():.4f}, Entropy: {entropy.item():.4f}")

def ppo_update_ad(data_buffer, ppo_epochs, clip_param):
    for _ in range(ppo_epochs):
        for data_batch in data_buffer:
            new_dist, new_value = ad_model(data_batch["states"], data_batch["time"], data_batch['z'])
            entropy = new_dist.entropy().mean()

            actor_loss = ppo_loss(new_dist, data_batch["actions"], data_batch["log_probs"], data_batch["advantages"],
                                  clip_param)

            critic_loss = clipped_critic_loss(new_value, data_batch["values"], data_batch["returns"], clip_param)
            agent_loss = critic_loss - actor_loss

            ad_optimizer.zero_grad()
            agent_loss.backward()
            nn.utils.clip_grad_norm_(ad_model.parameters(), 40)
            ad_optimizer.step()
        if _ == ppo_epochs - 1:  # Only log last epoch
            print(f"  Loss_adversary - Actor: {actor_loss.item():.4f}, Critic: {critic_loss.item():.4f}, Entropy: {entropy.item():.4f}")



# Training parameters
max_frames = 5e7
frames_seen = 0
rollouts = 0

# Score loggers
test_score_logger = []
train_score_logger = []
frames_logger = []


buffer_size = num_steps * num_envs
buffer_size_ad = num_steps_ad * num_envs
# Calculate the size of each minibatch  - usually very big - 2048!
mini_batch_size = buffer_size // num_mini_batch
mini_batch_size_ad = buffer_size_ad // num_mini_batch_ad
# Define the data we wish to collect for the databuffer
data_names = ["states", "actions", "log_probs", "values", "returns", "advantages", "directions"]
data_names_ad = ["states", "actions", "log_probs", "values", "returns", "advantages", "time",'z']

data_buffer = ReplayBuffer(data_names, buffer_size, mini_batch_size, device)
data_buffer_an = ReplayBuffer(data_names, buffer_size, mini_batch_size, device)
data_buffer_ad = ReplayBuffer(data_names_ad, buffer_size_ad, mini_batch_size_ad, device)


# Create the actor critic Model and optimizer
rl_model = Protagonist().to(device)
optimizer = optim.Adam(rl_model.parameters(), lr=lr)

ad_model = Adversary().to(device)
ad_optimizer = optim.Adam(ad_model.parameters(), lr=lr)

an_model = Antagonist().to(device)
an_optimizer = optim.Adam(an_model.parameters(), lr=lr)

env_fns = [make_env(agent=ad_model) for i in range(num_envs)]
envs = SyncVectorEnv(env_fns, autoreset_mode=AutoresetMode.DISABLED)
start_time = time.time()

antagonist_performance = []
protagonist_performance = []
while frames_seen < max_frames:
    rl_model.train()
    an_model.train()
    ad_model.train()
    # Initialise state
    start_state, _ = envs.reset()  #[30,7,7,3]
    if rollouts%15 == 0:
        first_env = envs.envs[0] 
        grid_img = first_env.render()   
        plt.imshow(grid_img)
        #saving to snap_shots folder
        plt.title(f"Grid at {frames_seen} frames")
        plt.savefig(f"./snap_shots/grid_at_{frames_seen}.png")
        plt.close()
    obs = start_state['image']
    direction = torch.tensor(start_state['direction'], dtype=torch.long).to(device)  # Shape: [num_envs]
    state = state_to_tensor(obs, device)



    all_envs_states = envs.get_attr('states_ad')
    all_envs_actions = envs.get_attr('actions_ad')
    all_envs_log_probs = envs.get_attr('log_probs_ad')
    all_envs_values = envs.get_attr('values_ad')
    all_envs_rewards = envs.get_attr('rewards_ad')
    all_envs_masks = envs.get_attr('masks_ad')
    all_envs_time = envs.get_attr('time_ad')
    all_envs_z = envs.get_attr('z_ad')

    env_rows = [torch.tensor(list(dq), dtype=torch.float32) for dq in all_envs_actions]
    temp = torch.stack(env_rows)
    temp = temp.T
    result = deque([temp[t].unsqueeze(1) for t in range(temp.shape[0])])
    actions_ad = result

    env_rows = [torch.tensor(list(dq), dtype=torch.float32) for dq in all_envs_log_probs]
    temp = torch.stack(env_rows)
    temp = temp.T
    result = deque([temp[t].unsqueeze(1) for t in range(temp.shape[0])])
    log_probs_ad = result

    env_rows = [torch.tensor(list(dq), dtype=torch.float32) for dq in all_envs_values]
    temp = torch.stack(env_rows)
    temp = temp.T
    result = deque([temp[t].unsqueeze(1) for t in range(temp.shape[0])])
    values_ad = result

    env_rows = [torch.tensor(list(dq), dtype=torch.float32) for dq in all_envs_rewards]
    temp = torch.stack(env_rows)
    temp = temp.T
    result = deque([temp[t].unsqueeze(1) for t in range(temp.shape[0])])
    rewards_ad = result

    env_rows = [torch.tensor(list(dq), dtype=torch.float32) for dq in all_envs_masks]
    temp = torch.stack(env_rows)
    temp = temp.T
    result = deque([temp[t].unsqueeze(1) for t in range(temp.shape[0])])
    masks_ad = result

    env_rows = [torch.tensor(list(dq), dtype=torch.float32) for dq in all_envs_time]
    temp = torch.stack(env_rows)
    temp = temp.T
    result = deque([temp[t].unsqueeze(1) for t in range(temp.shape[0])])
    time_ad = result

    # for states

    #for states
    #all_envs_states is a list of deques so
    env_rows = []
    for dq in all_envs_states:
        env_states = [s.squeeze(0).permute(2, 0, 1).float() if s.ndim == 4 else s.permute(2,0,1).float() for s in dq]
        env_rows.append(torch.stack(env_states))

    state_matrix = torch.stack(env_rows)
    state_matrix = state_matrix.transpose(0, 1)
    result_states = deque([state_matrix[t] for t in range(state_matrix.shape[0])])
    states_ad = result_states

    env_rows = []
    for dq in all_envs_z:
        env_z = [z.squeeze(0).float() for z in dq]
        env_rows.append(torch.stack(env_z))

    z_matrix = torch.stack(env_rows)
    z_matrix = z_matrix.transpose(0, 1)
    result_z = deque([z_matrix[t] for t in range(z_matrix.shape[0])])
    z_ad = result_z

    # Create data loggers - deques a bit faster than lists
    log_probs = deque()
    values = deque()
    states = deque()
    actions = deque()
    rewards = deque()
    masks = deque()
    directions = deque()

    log_probs_an = deque()
    values_an = deque()
    states_an = deque()
    actions_an = deque()
    rewards_an = deque()
    masks_an = deque()
    directions_an = deque()



    step = 0
    cnt = 0
    done = np.zeros(num_envs)
    print("Rollout!")
    with torch.no_grad():  # Don't need computational graph for roll-outs
        while step < num_steps:
            #  Masks so we can separate out multiple games in the same environment
            dist, value = rl_model(state, direction)  # Forward pass of actor-critic model
            action = dist.sample()  # Sample action from distribution

            # Take the next step in the env
            next_state, reward, termination, truncation, info = envs.step(action.cpu().numpy())
            done = np.logical_or(termination, truncation)

            for i in range(num_envs):
                if termination[i] or truncation[i]:
                    reset_obs, reset_info = envs.envs[i].reset(options={"keep_world": True})
                    next_state['image'][i] = reset_obs['image']
                    next_state['direction'][i] = reset_obs['direction']
                    envs._autoreset_envs[i] = False 

            cnt += (reward>0).sum()
            # Reset hidden states for environments that finished
            # done is a numpy array of shape (num_envs,) with True/False values
            
            # Log data
            reward = torch.tensor(reward, dtype=torch.float32).to(device)
            log_prob = dist.log_prob(action)
            log_probs.append(log_prob)
            states.append(state)  # [num_steps,num_envs,3,7,7]
            actions.append(action)
            values.append(value)
            rewards.append(reward.unsqueeze(1).to(device))
            current_mask = torch.FloatTensor(1 - done).unsqueeze(1).to(device)
            masks.append(current_mask)
            directions.append(direction)

            direction = torch.tensor(next_state['direction'], dtype=torch.long).to(device)  # Shape: [num_envs]
            next_state = next_state['image']    
            state = state_to_tensor(next_state, device)
            step += 1
        
        print(f"protagonist: {cnt}")
        protagonist_performance.append(cnt)
        cnt = 0
        step = 0
        # start_state, _ = envs.reset(options = {"keep_world": True})  #[30,7,7,3]
        start_state = {'image': np.zeros((num_envs, 10, 10, 3), dtype=np.uint8), 'direction': np.zeros((num_envs,), dtype=np.int64)}
        for i, env in enumerate(envs.envs):
            obss, infoss = env.reset(options={"keep_world": True})
            start_state['image'][i] = obss['image']
            start_state['direction'][i] = obss['direction']

        obs = start_state['image']
        direction_an = torch.tensor(start_state['direction'], dtype=torch.long).to(device)  # Shape: [num_envs]
        state_an = state_to_tensor(obs, device)


        while step < num_steps:
            #  Masks so we can separate out multiple games in the same environment
            dist, value = an_model(state_an, direction_an)  # Forward pass of actor-critic model
            action = dist.sample()  # Sample action from distribution

            # Take the next step in the env
            next_state_an, reward_an, termination_an, truncation_an, info_an = envs.step(action.cpu().numpy())
            done = np.logical_or(termination_an, truncation_an)

            for i in range(num_envs):
                if termination_an[i] or truncation_an[i]:
                    reset_obs, reset_info = envs.envs[i].reset(options={"keep_world": True})
                    next_state_an['image'][i] = reset_obs['image']
                    next_state_an['direction'][i] = reset_obs['direction']
                    envs._autoreset_envs[i] = False 

            cnt += (reward_an>0).sum()
            # Reset hidden states for environments that finished
            # done is a numpy array of shape (num_envs,) with True/False values
            
            # Log data
            reward_an = torch.tensor(reward_an, dtype=torch.float32).to(device)
            log_prob = dist.log_prob(action)
            log_probs_an.append(log_prob)
            states_an.append(state_an)  # [num_steps,num_envs,3,7,7]
            actions_an.append(action)
            values_an.append(value)
            rewards_an.append(reward_an.unsqueeze(1).to(device))
            current_mask = torch.FloatTensor(1 - done).unsqueeze(1).to(device)
            masks_an.append(current_mask)
            directions_an.append(direction_an)

            direction_an = torch.tensor(next_state_an['direction'], dtype=torch.long).to(device)  # Shape: [num_envs]
            next_state_an = next_state_an['image']
            state_an = state_to_tensor(next_state_an, device)
            step += 1
        
        #  regret calculation and updating adversary reward

        ant = torch.stack([r.squeeze(-1) for r in rewards_an])  #(steps,num_env)
        prot = torch.stack([r.squeeze(-1) for r in rewards])
        # ant_max = ant.max(dim=0).values # (num_env)
        mask = prot != 0 #(steps,num_env)
        mask1 = ant != 0
        prot_sum = prot.sum(dim=0) #(32)
        ant_sum = ant.sum(dim=0)
        prot_count = mask.sum(dim=0) #(32)
        ant_count = mask1.sum(dim=0)
        prot_mean = torch.where(prot_count > 0, prot_sum / prot_count, torch.zeros_like(prot_sum))
        ant_mean = torch.where(ant_count > 0, ant_sum / ant_count, torch.zeros_like(ant_sum))
        regret = torch.clamp(ant_mean - prot_mean, min=0).unsqueeze(-1)
        rewards_ad[-1] = regret

        print(f"REGRET STATISTICS (Rollout {rollouts}):")
        print(f"  Antagonist max:  {ant_mean.mean():.3f} ± {ant_mean.std():.3f} | range [{ant_mean.min():.3f}, {ant_mean.max():.3f}]")
        print(f"  Protagonist mean: {prot_mean.mean():.3f} ± {prot_mean.std():.3f} | range [{prot_mean.min():.3f}, {prot_mean.max():.3f}]")
        print(f"  Regret:          {regret.mean():.3f} ± {regret.std():.3f} | range [{regret.min():.3f}, {regret.max():.3f}]")

        print(f"Antagonist: {cnt}")
        antagonist_performance.append(cnt)
        # tala remain
        # adversary ko buffer + reward = regret garna banki
        # 3 ota kai ppo update farak
        # tespaxi necessary savings, necessary informations savings
        # evolving state shape, rewards, progress, 

        # Get value at time step T+1
        _, next_value = rl_model(state, direction)
        _, next_value_an = an_model(state_an, direction_an)
        next_value_ad = torch.tensor([0.0] * num_envs).to(device)   # because we always reach end
        next_value_ad = next_value_ad.unsqueeze(1)
        # Calculate the returns/gae
        returns, advantage = compute_gae(next_value, rewards, masks, values, gamma=gamma, tau=tau)


        data_buffer.data_log("states", torch.cat(list(states)))
        data_buffer.data_log("actions", torch.cat(list(actions)))
        data_buffer.data_log("returns", torch.cat(list(returns)))
        data_buffer.data_log("log_probs", torch.cat(list(log_probs)))
        data_buffer.data_log("values", torch.cat(list(values)))
        data_buffer.data_log("directions", torch.cat(list(directions)))
        advantage = torch.cat(list(advantage)).squeeze(1)
        data_buffer.data_log("advantages", (advantage - advantage.mean()) / (advantage.std() + 1e-8))



        # antagonist
        returns_an, advantage_an = compute_gae(next_value_an, rewards_an, masks_an, values_an, gamma=gamma, tau=tau)
        data_buffer_an.data_log("states", torch.cat(list(states_an)))
        data_buffer_an.data_log("actions", torch.cat(list(actions_an)))
        data_buffer_an.data_log("returns", torch.cat(list(returns_an)))
        data_buffer_an.data_log("log_probs", torch.cat(list(log_probs_an)))
        data_buffer_an.data_log("values", torch.cat(list(values_an)))
        data_buffer_an.data_log("directions", torch.cat(list(directions_an)))
        advantage_an = torch.cat(list(advantage_an)).squeeze(1)
        data_buffer_an.data_log("advantages", (advantage_an - advantage_an.mean()) / (advantage_an.std() + 1e-8))



        #adversary
        returns_ad, advantage_ad = compute_gae(next_value_ad, rewards_ad, masks_ad, values_ad, gamma=gamma, tau=tau)
        data_buffer_ad.data_log("states", torch.cat(list(states_ad)))
        data_buffer_ad.data_log("actions", torch.cat(list(actions_ad)))
        data_buffer_ad.data_log("returns", torch.cat(list(returns_ad)))
        data_buffer_ad.data_log("log_probs", torch.cat(list(log_probs_ad)))
        data_buffer_ad.data_log("values", torch.cat(list(values_ad)))
        data_buffer_ad.data_log("time", torch.cat(list(time_ad)))
        data_buffer_ad.data_log("z", torch.cat(list(z_ad)))

        advantage_ad = torch.cat(list(advantage_ad)).squeeze(1)
        data_buffer_ad.data_log("advantages", (advantage_ad - advantage_ad.mean()) / (advantage_ad.std() + 1e-8))

        # Update the frames counter
        # We normaly base how long to train for by counting the number of "environment interactions"
        # In our case we can simply counte how many game frames we have received from the environment
        frames_seen += advantage.shape[0]
    
    # We train after every batch of rollouts
    # With the stabalisation techniques in PPO we can "safely" take many steps with a single
    # batch of rollouts, therefore we usualy train with the data over multiple epochs whereas basic
    # actor critic methods only use one epoch.
    # Before updating each agent
    print(f"\nVALUE PREDICTIONS:")
    print(f"  Protagonist values:  {torch.cat(list(values)).mean():.3f} ± {torch.cat(list(values)).std():.3f}")
    print(f"  Antagonist values:   {torch.cat(list(values_an)).mean():.3f} ± {torch.cat(list(values_an)).std():.3f}")
    print(f"  Adversary values:    {torch.cat(list(values_ad)).mean():.3f} ± {torch.cat(list(values_ad)).std():.3f}")

    # Before updating each agent
    print(f"\nVALUE PREDICTIONS:")
    print(f"  Protagonist values:  {torch.cat(list(values)).mean():.3f} ± {torch.cat(list(values)).std():.3f}")
    print(f"  Antagonist values:   {torch.cat(list(values_an)).mean():.3f} ± {torch.cat(list(values_an)).std():.3f}")
    print(f"  Adversary values:    {torch.cat(list(values_ad)).mean():.3f} ± {torch.cat(list(values_ad)).std():.3f}")
    print("Training")
    ppo_update(data_buffer, ppo_epochs, clip_param)
    ppo_update_an(data_buffer_an, ppo_epochs, clip_param)
    ppo_update_ad(data_buffer_ad, ppo_epochs, clip_param)
    seed_counter.seed_count += 30
    rollouts += 1
    if rollouts % 1 == 0:
        # print("Testing")
        rl_model.eval()
        # TODO: Implement testing for custom environment
        # test_score = evaluate_agent(env, rl_model, device)
        # train_score = run_tests(train_test="train")
        # test_score = 0  # Placeholder
        # train_score = 0  # Placeholder

        # test_score_logger.append(test_score)
        # frames_logger.append(frames_seen)
        # print("Trained on %d Frames, Test Score [%d/%d]" 
        #     %(frames_seen, test_score))
        if rollouts%200 == 0:
            # save protagonist and antagonist performance graph too
            plt.figure(figsize=(12,5))
            plt.subplot(1,2,1)
            plt.title('Protagonist Performance')
            plt.plot(protagonist_performance, label='Protagonist Performance')
            plt.xlabel('Rollouts')
            plt.ylabel('Performance')
            plt.legend()
            plt.subplot(1,2,2)
            plt.title('Antagonist Performance')
            plt.plot(antagonist_performance, label='Antagonist Performance')
            plt.xlabel('Rollouts')
            plt.ylabel('Performance')
            plt.legend()
            plt.savefig(f'./saved_figures/performance_{rollouts}.png')
            plt.close()
            torch.save(rl_model.state_dict(), f"./saved_models/Protagonist_{rollouts}.pth")
            torch.save(an_model.state_dict(), f"./saved_models/Antagonist_{rollouts}.pth")
            torch.save(ad_model.state_dict(), f"./saved_models/Adversary_{rollouts}.pth")
        print("Trained on %d Frames" 
            %(frames_seen))
        time_to_end = ((time.time() - start_time) / frames_seen) * (max_frames - frames_seen)
        print("Time to end: %dh:%dm" % (time_to_end // 3600, (time_to_end % 3600) / 60))



torch.save(rl_model.state_dict(), f"Final_Protagonist.pth")
torch.save(an_model.state_dict(), f"Final_Antagonist.pth")
torch.save(ad_model.state_dict(), f"Final_Adversary.pth")

print(f" training finsihed for seed. saved!")

print("Done!")

#1. protagonist ko architecture change because, now we have 30 env
#2. Ani ho reset vayepaxi, hidden k hunxa? (jun env jaile ni reset hunu sakxa, hidden ta zero chiyo initially)