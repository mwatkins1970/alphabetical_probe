class MultiDimensionalProbe(nn.Module):
    def __init__(self, input_dim, output_dim=2):  
        super(MultiDimensionalProbe, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):             
        return self.fc(x)

def multi_dim_loss(outputs, labels):
    # Compute dot product
    dot_product = torch.sum(outputs * labels, dim=-1)
    
    # Apply sigmoid
    probs = torch.sigmoid(dot_product)
    
    # Compute binary cross-entropy loss
    criterion = nn.BCELoss()
    return criterion(probs, labels)