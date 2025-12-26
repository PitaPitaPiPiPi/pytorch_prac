import torch.nn as nn
import torch.nn.functional as F

class MLP5Layer(nn.Module):
    def __init__(self, input_size=784, hidden_sizes=[512, 256, 128, 64], num_classes=10):
        super(MLP5Layer, self).__init__()
        self.flatten = nn.Flatten()
        
        layers = []
        in_dim = input_size
        for h_dim in hidden_sizes:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim
        
        self.hidden_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        x = self.flatten(x)
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        return x