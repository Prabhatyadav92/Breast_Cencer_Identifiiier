import torch
import torch.nn as nn

class BreastCancerModel(nn.Module):
    def __init__(self, input_features):
        super(BreastCancerModel, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_features, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()   # Binary classification
        )

    def forward(self, x):
        return self.model(x)
