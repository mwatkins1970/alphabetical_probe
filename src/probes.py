import torch.nn as nn

class LinearProbe(nn.Module):
    def __init__(self, input_dim):                    # constructor method, called automaticaly when a class instance is called
        super().__init__()
        self.fc = nn.Linear(input_dim, 1)  # Define a Fully Connected linear layer with 1 output neuron.

    def forward(self, x):             # implicitly used in model(batch_embeddings) below, which calls the 'forward' method of the 'model' object
        return self.fc(x)

