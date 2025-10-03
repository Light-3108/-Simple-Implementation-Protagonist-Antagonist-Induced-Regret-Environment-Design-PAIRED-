import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
# def set_seed(seed=42):
#     random.seed(seed)                      
#     np.random.seed(seed)                   
#     torch.manual_seed(seed)                
#     torch.cuda.manual_seed(seed)           
#     torch.cuda.manual_seed_all(seed)       
#     torch.backends.cudnn.deterministic = True  
#     torch.backends.cudnn.benchmark = False     

# set_seed(123)  

class Adversary(nn.Module):
    def __init__(self, num_actions=64, conv_filters=64, fc_hidden=256):
        super(Adversary, self).__init__()
        
        # 1st convolution 
        self.conv = nn.Conv2d(in_channels=3, out_channels=conv_filters, kernel_size=3)  # (batch, conv_filters, 8, 8)

        # Timestep embedding
        self.t_fc = nn.Linear(1, 10)

        # Fully connected after concatenation of conv features + timestep embedding + random vector z
        conv_out_size = conv_filters * 8 * 8
        self.fc_combined = nn.Linear(conv_out_size + 10 + 10, fc_hidden)

        # Additional FC layers
        self.fc1 = nn.Linear(fc_hidden, 32)
        self.fc2 = nn.Linear(32, 32)

        # Policy head
        self.policy = nn.Linear(32, num_actions)

        # Value head
        self.value = nn.Linear(32, 1)

    def forward(self, obs, t, z):

        batch_size = obs.size(0)

        # Conv layers
        x = obs
        # x = obs.permute(0,3,1,2)  # (1,10,10,3)
        x = F.relu(self.conv(x))          # (batch, conv_filters, 8, 8)
        x = x.reshape(batch_size, -1)     # flatten → (batch, conv_filters*8*8)

        # Timestep embedding
        t_emb = F.relu(self.t_fc(t))      # (batch, 10)

        # Concatenate with random vector z
        combined = torch.cat([x, t_emb, z], dim=-1)  # (batch, conv+10+50)

        # Fully connected after concatenation
        h = F.relu(self.fc_combined(combined))

        # Additional FC layers
        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))

        # Outputs
        policy_logits = self.policy(h)  # (batch, num_actions)
        value = self.value(h)           # (batch, 1)

        dist = Categorical(logits=policy_logits)
        return dist, value
