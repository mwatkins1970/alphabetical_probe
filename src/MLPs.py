import torch.nn as nn
import torch.nn.functional as F

class MLPProbe(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_hidden_layers):
        super(MLPProbe, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        
        # Input Layer
        self.fc_input = nn.Linear(input_dim, hidden_dim)
        
        # Hidden Layers
        self.fc_hidden = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_hidden_layers - 1)]
        )
        
        # Output Layer
        self.fc_output = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        x = F.relu(self.fc_input(x))
        for fc in self.fc_hidden:
            x = F.relu(fc(x))
        x = self.fc_output(x)
        return x
