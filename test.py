import torch
import matplotlib.pyplot as plt
from minigrid.wrappers import FullyObsWrapper
# from environments.out_of_dist import make_env


from protagonist_mdp import Protagonist
from antagonist import Antagonist
from gymnasium.vector import SyncVectorEnv, AsyncVectorEnv
from environments.four_rooms import make_env
from helper import *
import imageio

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Antagonist(num_actions=3).to(device)
model.load_state_dict(torch.load("Antagonist_1200.pth", map_location=device))
model.eval()


num_envs = 1
env = SyncVectorEnv([make_env() for _ in range(num_envs)])
# env = SimpleEnv(render_mode="rgb")
# env = FullyObsWrapper(env)
frames = []
# # Get full observations
# def state_to_tensor(obs, device="cpu"):  # [7,7,3]
#     obs = torch.tensor(obs, dtype=torch.float32, device=device)
#     obs = obs.permute(2, 0, 1)  # → [3,7,7]
#     return obs

episodes = 50
episode_rewards = []


for ep in range(episodes):
    obs, _ = env.reset()
    direction = torch.tensor(obs['direction'], dtype=torch.long).to(device)  # Shape: [num_envs]
    obs = obs['image']

    done = False
    total_reward = 0
    while not done:
        state = state_to_tensor(obs, device)  # [1,3,10,10]

        dist, _ = model(state, direction)

        # # Option 1: deterministic (greedy)
        action = dist.probs.argmax(dim=1).item()

        # # Option 2: stochastic (explores when policy uncertain)
        # action = dist.sample().item()

        obs, reward, terminated, truncated, _ = env.step([action])
        done = terminated[0] or truncated[0]
        total_reward += reward[0]
        # update direction and obs
        direction = torch.tensor(obs['direction'], dtype=torch.long).to(device)  # Shape: [num_envs]
        obs = obs['image']
        frame = env.render()
        frame = frame[0]
        frame = np.squeeze(frame)
        frames.append(frame)

    episode_rewards.append(total_reward)
    print(f"Episode {ep+1} Reward: {total_reward}")

env.close()

# imageio.mimsave("2_rooms.gif", frames, fps=5, loop=0)

# mean_reward = np.mean(episode_rewards)
# no_solved = np.sum(np.array(episode_rewards) > 0)

# print(mean_reward,no_solved)
# # === Plot rewards ===
# plt.plot(range(1, episodes + 1), episode_rewards, marker='o')
# plt.xlabel("Episode")
# plt.ylabel("Reward")
# plt.title("Evaluation Rewards")
# plt.show()
