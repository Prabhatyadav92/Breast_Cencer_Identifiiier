import torch
import torch.nn as nn

class BreastCancerModel(nn.Module):
    def __init__(self, input_features):
        super(BreastCancerModel, self).__init__()

        self.fc1 = nn.Linear(input_features, 16)
        self.fc2 = nn.Linear(16, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x
