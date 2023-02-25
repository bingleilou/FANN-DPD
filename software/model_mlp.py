# coding: utf-8
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

# Set the random seed manually for reproducibility.
seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


class Full_Connect_MLP(nn.Module):
    def __init__(self, input_sizes=8, output_sizes=8, hidden_size=8):
        super(Full_Connect_MLP, self).__init__()
        
        self.input_sizes = input_sizes
        self.output_sizes = output_sizes

        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(input_sizes,hidden_size)
        self.fc2 = nn.Linear(hidden_size,hidden_size*2)
        self.fc3 = nn.Linear(hidden_size*2,hidden_size)
        self.fc4 = nn.Linear(hidden_size,output_sizes)

    def forward(self, x):
        x = x.view(-1, self.input_sizes)
        x_short = torch.index_select(x, 1, torch.tensor([int(self.input_sizes/2)-1,int(self.input_sizes)-1]))
        x_short = x_short.view(-1, 2)
        
        a1 = self.fc1(x)
        z1 = self.relu(a1)

        a2 = self.fc2(z1)
        z2 = self.relu(a2)

        a3 = self.fc3(z2)
        z3 = self.relu(a3)

        out = self.fc4(z3) + x_short

        return out
     

