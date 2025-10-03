# Protagonist architecture for MiniGrid

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class Protagonist(nn.Module):
    def __init__(self, num_actions=3):
        super(Protagonist, self).__init__()

        # CNN feature extractor
        self.conv_net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1),  # (batch, 16, 8, 8)
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1), # (batch, 32, 6, 6)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1), # (batch, 64, 4, 4)
            nn.ReLU(),
            nn.Flatten(),                               # (batch, 64*4*4 = 1024)
        )

        # Direction encoding
        self.dir_fc = nn.Linear(4, 16)

        # Fully connected layers
        self.fc1 = nn.Linear(1024 + 16, 256)
        self.fc2 = nn.Linear(256, 128)

        # Policy & Value heads
        self.policy = nn.Linear(128, num_actions)
        self.value = nn.Linear(128, 1)

    def forward(self, obs, direction):
        """
        obs: tensor (batch, 3, 10, 10)
        direction: tensor (batch,) with values 0..3
        hidden: unused (kept for API compatibility)
        """
        batch_size = obs.size(0)

        # CNN feature extraction
        x = self.conv_net(obs)

        # Direction one-hot
        dir_onehot = F.one_hot(direction, num_classes=4).float()
        d = F.relu(self.dir_fc(dir_onehot))

        # Concatenate CNN + direction
        features = torch.cat([x, d], dim=-1)

        # Fully connected
        h = F.relu(self.fc1(features))
        h = F.relu(self.fc2(h))

        # Outputs
        policy_logits = self.policy(h)
        value = self.value(h)

        dist = Categorical(logits=policy_logits)
        return dist, value
