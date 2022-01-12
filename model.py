import torch # Library for deep learning developed and maintained by Facebook.
import torch.nn as nn # Base class used to develop all neural network models.

"""
model.py: A feed forward hidden neural net with two hidden layers
"""

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        # Three linear layers:
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size) # First layer
        self.l2 = nn.Linear(hidden_size, hidden_size) # Second layer
        self.l3 = nn.Linear(hidden_size, num_classes) # Third layer
        self.relu = nn.ReLU() # Actuation function

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        return out