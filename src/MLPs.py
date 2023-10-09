import torch.nn as nn
import torch.nn.functional as F

class MLPProbe(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_hidden_layers):
        super(MLPProbe, self).__init__()
        assert num_hidden_layers >= 2, "Number of hidden layers should be at least 2 for this architecture"
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        
        # Input Layer
        self.fc_input = nn.Linear(input_dim, hidden_dim)
        
        # Hidden Layers
        self.fc_hidden = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_hidden_layers - 1)]
        )
        
        # Dropout
        self.dropout = nn.Dropout(0.1)
        
        # Output Layer
        self.fc_output = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # First hidden layer with SELU activation
        x = F.selu(self.fc_input(x))
        
        # Second hidden layer with Tanh activation
        x = F.tanh(self.fc_hidden[0](x))
        
        # If more than 2 hidden layers, apply ReLU for the rest
        for fc in self.fc_hidden[1:]:
            x = F.relu(fc(x))
        
        # Apply dropout
        x = self.dropout(x)
        
        # Output layer
        x = self.fc_output(x)
        
        return x
