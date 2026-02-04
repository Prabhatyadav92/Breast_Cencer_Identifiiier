import torch
import torch.nn as nn

class BreastCancerClassifier(nn.Module):
    def __init__(self, input_features, hidden_features, output_features):
        super(BreastCancerClassifier, self).__init__()

        self.fc1 = nn.Linear(input_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, output_features)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
