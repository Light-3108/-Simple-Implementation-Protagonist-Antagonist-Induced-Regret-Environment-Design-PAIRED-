import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class Adversary(nn.Module):
    def __init__(self, num_actions=64):
        super(Adversary, self).__init__()

        # CNN feature extractor (matching antagonist depth)
        self.conv_net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1),  # (batch, 16, 8, 8)
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1), # (batch, 32, 6, 6)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1), # (batch, 64, 4, 4)
            nn.ReLU(),
            nn.Flatten(),                               # (batch, 1024)
        )

        # Timestep encoding (expanded)
        self.t_fc = nn.Linear(1, 32)
        
        # z embedding (for stochasticity, size 20)
        self.z_fc = nn.Linear(20, 32)

        # Fully connected layers (matching antagonist structure)
        self.fc1 = nn.Linear(1024 + 32 + 32, 256)  # CNN + timestep + z
        self.fc2 = nn.Linear(256, 128)

        # Policy & Value heads
        self.policy = nn.Linear(128, num_actions)
        self.value = nn.Linear(128, 1)

    def forward(self, obs, t, z):
        """
        obs: tensor (batch, 3, 10, 10)
        t: tensor (batch, 1) - timestep
        z: tensor (batch, 20) - random vector for stochasticity
        """
        batch_size = obs.size(0)

        # CNN feature extraction
        x = self.conv_net(obs)

        # Timestep embedding
        t_emb = F.relu(self.t_fc(t))
        
        # z embedding
        z_emb = F.relu(self.z_fc(z))

        # Concatenate CNN + timestep + z
        features = torch.cat([x, t_emb, z_emb], dim=-1)

        # Fully connected
        h = F.relu(self.fc1(features))
        h = F.relu(self.fc2(h))

        # Outputs
        policy_logits = self.policy(h)
        value = self.value(h)

        dist = Categorical(logits=policy_logits)
        return dist, value