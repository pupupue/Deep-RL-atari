import torch
import torch.nn as nn
import torch.nn.functional as F

class NNet(nn.Module):
    def __init__(self, input_shape, output_shape, hidden_shape, device=torch.device("cpu")):
        """
        :param input_shape: Shape/dimension of the input
        :param output_shape: Shape/dimension of the output
        :param device: The device (cpu or cuda) that the SLP should use to store the inputs for the forward pass
        """
        super(NNet, self).__init__()
        self.device = device
        self.input_shape = input_shape[0]
        self.hidden_shape = hidden_shape
        self.output_shape = output_shape
        self.l1 = nn.Linear(self.input_shape, self.hidden_shape)
        self.l2 = nn.Linear(self.hidden_shape, self.output_shape)

    def forward(self, x):
        x = torch.from_numpy(x).float().to(self.device)
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x