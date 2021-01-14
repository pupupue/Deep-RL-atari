import torch
import torch.nn as nn
import torch.nn.functional as F

class CONV(nn.Module):
    def __init__(self, input_shape, output_shape, device="cpu", hidden_shape=256):
        # input_shape: C x 84 x 84
        super(CONV, self).__init__()
        self.device = device
        self.poolavg = nn.AvgPool2d(2, 2)
        self.dropout = nn.Dropout(p=0.1)
        self.conv1 = nn.Conv2d(input_shape[0], 64, kernel_size=8, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=0)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=2, stride=1, padding=0)
        self.l1 = nn.Linear(128 * 3 * 3, hidden_shape)
        self.l2 = nn.Linear(hidden_shape, output_shape)

    def forward(self, x):
        x = torch.from_numpy(x).float().to(self.device) # input: torch.Size([1, 4, 84, 84])
        x = F.leaky_relu(self.conv1(x)) # cov1: torch.Size([1, 64, 40, 40])
        x = self.poolavg(x) # pool: torch.Size([1, 64, 20, 20])
        x = F.leaky_relu(self.conv2(x)) # cov2: torch.Size([1, 64, 9, 9])
        x = self.poolavg(x) # cov2: torch.Size([1, 64, 5, 5])
        x = F.leaky_relu(self.conv3(x)) # cov3: torch.Size([1, 128, 3, 3])
        x = x.view(x.shape[0], -1)
        x = self.dropout(F.leaky_relu(self.l1(x)))
        x = self.l2(x)
        return x
        # 

'''
input: torch.Size([1, 4, 84, 84])
cov1: torch.Size([1, 64, 40, 40])
pool: torch.Size([1, 64, 20, 20])
cov2: torch.Size([1, 64, 9, 9])
cov3: torch.Size([1, 128, 3, 3])
reshape: torch.Size([1, 1152])
Lin1: torch.Size([1, 256])
Lin1: torch.Size([1, 18])

# single
def forward(self, x):
    x = self.dropout(F.leaky_relu(self.l1(x)))
    x = self.l2(x)
    return x


# debug
# in self.out(x*y*z) are breaks

x = torch.from_numpy(x).float().to(self.device)
print(f"input: {x.shape}")

x = F.leaky_relu(self.conv1(x))
print(f"cov1: {x.shape}")

x = self.poolavg(x)
print(f"pool: {x.shape}")

x = F.leaky_relu(self.conv2(x))
print(f"cov2: {x.shape}")

x = self.poolavg(x)

x = F.leaky_relu(self.conv3(x))
print(f"cov3: {x.shape}")

x = x.view(x.shape[0], -1)
print(f"reshape: {x.shape}")

x = self.dropout(F.leaky_relu(self.l1(x)))
print(f"Lin1: {x.shape}")

x = self.l2(x)
print(f"Lin1: {x.shape}")
return x
'''
