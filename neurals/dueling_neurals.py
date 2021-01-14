import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
import numpy as np
"""
DUELING ESTIMATOR NETS:
    @ Value
    @ Advantage
"""

"""
VALUE ESTIMATE LAYER:
input_shape = last layer of conv { torch.Size([1, 1152]) }
output_shape = 1
"""
class Value_estimate(nn.Module):

    def __init__(self, input_shape, device, output_shape=1, hidden_shape=128):
        super(Value_estimate, self).__init__()
        self.device = device
        self.dropout = nn.Dropout(p=0.01)
        self.input_shape = input_shape
        self.hidden_shape = hidden_shape
        self.output_shape = output_shape
        self.l1 = nn.Linear(self.input_shape, self.hidden_shape).to(self.device)
        self.l2 = nn.Linear(self.hidden_shape, self.output_shape).to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        x = self.dropout(F.relu(self.l1(x)))
        x = self.l2(x)
        return x

"""
ADVANTAGE ESTIMATE LAYER:
input_shape = last layer of conv { torch.Size([1, 1152]) }
output_shape = (6) [action_size] 
"""
class Advantage_estimate(nn.Module):

    def __init__(self, input_shape, output_shape, device, hidden_shape=128):
        super(Advantage_estimate, self).__init__()
        self.device = device
        self.dropout = nn.Dropout(p=0.01)
        self.input_shape = input_shape
        self.hidden_shape = hidden_shape
        self.output_shape = output_shape
        self.l1 = nn.Linear(self.input_shape, self.hidden_shape).to(self.device)
        self.l2 = nn.Linear(self.hidden_shape, self.output_shape).to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        x = self.dropout(F.leaky_relu(self.l1(x)))
        x = self.l2(x)
        return x


'''
input: torch.Size([1, 4, 84, 84])
cov1: torch.Size([1, 64, 40, 40])
pool: torch.Size([1, 64, 20, 20])
cov2: torch.Size([1, 64, 9, 9])
cov3: torch.Size([1, 128, 3, 3])
reshape: torch.Size([1, 1152])
Lin1: torch.Size([1, 256])
Lin1: torch.Size([1, 18])
'''
class CONV(nn.Module):
    def __init__(self, input_shape, device):
        # input_shape: C x 84 x 84
        super(CONV, self).__init__()
        self.device = device
        self.input_shape = input_shape
        self.poolavg = nn.AvgPool2d(2, 2)
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4, padding=0).to(self.device)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0).to(self.device)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0).to(self.device)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x)) # cov1: torch.Size([1, 64, 40, 40])
        x = self.poolavg(x) # pool: torch.Size([1, 64, 20, 20])
        x = F.relu(self.conv2(x)) # cov2: torch.Size([1, 64, 9, 9])
        x = F.relu(self.conv3(x)) # cov3: torch.Size([1, 128, 3, 3])
        x = x.view(x.shape[0], -1) # reshape: torch.Size([1, 1152])
        return x
    
    def get_last_layers(self):        
        x = np.zeros(self.input_shape, dtype=np.float32)
        x = np.expand_dims(x, axis=0)
        x = torch.from_numpy(x).float().to(self.device)
        res = self.forward(x)
        res = [int(x) for x in res[0].shape]
        return res[0] # int


'''
[D]ueling[D]eep[Q][N]etwork:

'''
class DDQN(nn.Module):
    def __init__(self, state_shape, action_shape, device="cpu"):
        super(DDQN, self).__init__()
        self.device = device
        self.conv = CONV(state_shape, device)
        self.A = Advantage_estimate(self.conv.get_last_layers(), action_shape, device)
        self.V = Value_estimate(self.conv.get_last_layers(), device)
    
    def forward(self, x, test=False):
        x = torch.from_numpy(x).float().to(self.device) # input: torch.Size([1, 4, 84, 84])
        if test:
            x = x.detach()
            self.conv.eval()
            self.V.eval()
            self.A.eval()
        
        x = self.conv(x).to(self.device)
        V = self.V(x).to(self.device)
        A = self.A(x).to(self.device)
        
        Q = V + (A - A.mean())
        return Q


if __name__ == "__main__":
    state_shape = (4, 84, 84) 
    action_shape = 6

    # x = np.zeros(state_shape, dtype=np.float32)
    # dqn = DDQN(state_shape, action_shape)
    # # print(x.shape)
    # dqn(x)
    conv = CONV(state_shape)
    conv.get_last_layers()