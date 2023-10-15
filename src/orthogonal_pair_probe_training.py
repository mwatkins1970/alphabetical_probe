import torch 
import torch.nn as nn 
import torch.optim as optim
import wandb 
import numpy as np
import tempfile
import os

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# Import adjustments for Colab vs local environments
try:
    from src.dataset import LetterDataset
except ImportError:
    from dataset import LetterDataset

try:
    from src.probes import LinearProbe
except ImportError:
    from probes import LinearProbe

class OrthogonalProbePair(nn.Module):
    """ 
    A model representing two linear probes.
    """
    def __init__(self, input_dim):
        super(OrthogonalProbePair, self).__init__()
        self.probe1 = nn.Linear(input_dim, 1)
        self.probe2 = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.probe1(x), self.probe2(x)

# Set random seeds for reproducibility
rnd_seed = 42
torch.manual_seed(rnd_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(rnd_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
np.random.seed(rnd_seed)

def orthogonal_constraint(probe1_weights, probe2_weights):
    """ 
    Computes the loss term encouraging orthogonality between the two probe weights.
    Squares the dot product to amplify the deviation from orthogonality.
    """
    dot_product = torch.sum(probe1_weights * probe2_weights)
    return dot_product**2

def create_and_log_artifact(tensor, name, artifact_type, description):
    """
    Logs a tensor as a Weights & Biases artifact.
    """
    with tempfile.NamedTemporaryFile(delete=True, suffix='.pt') as tmp:
        torch.save(tensor, tmp.name)
        
        artifact = wandb.Artifact(
            name=name,
            type=artifact_type,
            description=description,
        )
        artifact.add_file(tmp.name)
        wandb.log_artifact(artifact)

def train_letter_probe_runner(letter, embeddings, token_strings, all_rom_token_indices,
                              num_samples, num_epochs, batch_size, device, criteria_mode,
                              use_wandb=False, wandb_group_name=None, learning_rate=0.001,
                              orthogonality_weight=1.0):
    """
    Trains a pair of orthogonal probes for detecting a specific letter.
    """
    if use_wandb:
        config = {
            "letter": letter,            
            "criteria_mode": criteria_mode,
            "model_name": "gpt-j",
            "probe_type": probe_type,
            "train_test_split": 0.2,
            "seed": rnd_seed,
            "case_sensitive": False,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "num_samples": num_samples,
            "num_epochs": num_epochs,
            "device": device,
        }
        wandb.init(
            project="letter_presence_probes",
            group=wandb_group_name,
            config=config,
        )

    # Train-test split logic
    X, y = embeddings[:num_samples], token_strings[:num_samples]  # Assuming token_strings contain labels. Adjust accordingly.
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=rnd_seed)
    
    embeddings_dim = embeddings.shape[1]
    
    # Initialize the OrthogonalProbePair model
    model = OrthogonalProbePair(embeddings_dim).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    
    train_dataset = LetterDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for batch_embeddings, batch_labels in train_loader:
            batch_embeddings = batch_embeddings.to(device).float()
            batch_labels = batch_labels.to(device).float()

            optimizer.zero_grad()
            outputs_probe1, outputs_probe2 = model(batch_embeddings)
            loss_probe1 = criterion(outputs_probe1.squeeze(), batch_labels)
            loss_probe2 = criterion(outputs_probe2.squeeze(), batch_labels)
            
            orthogonality_loss = orthogonal_constraint(model.probe1.weight, model.probe2.weight) * orthogonality_weight

            total_loss = loss_probe1 + loss_probe2 + orthogonality_loss
            total_loss.backward()
            optimizer.step()

            if use_wandb:
                wandb.log({"loss": total_loss.item()})
                
        print(f"{letter}: epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader)}")

        # Validation phase
        model.eval()
        val_dataset = LetterDataset(X_val, y_val)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        correct_preds = 0
        total_preds = 0
        validation_loss = 0.0
        all_labels, all_predictions = [], []

        with torch.no_grad():
            for batch_embeddings, batch_labels in val_loader:
                batch_embeddings = batch_embeddings.to(device).float()
                batch_labels = batch_labels.to(device).float()

                outputs_probe1, outputs_probe2 = model(batch_embeddings)
                
                # Compute loss and accuracy
                loss_probe1 = criterion(outputs_probe1.squeeze(), batch_labels)
                loss_probe2 = criterion(outputs_probe2.squeeze(), batch_labels)
                validation_loss += (loss_probe1 + loss_probe2).item()
                
                preds = (torch.sigmoid(outputs_probe1) + torch.sigmoid(outputs_probe2)) / 2
                preds = preds > 0.5
                correct_preds += (preds.squeeze() == batch_labels).sum().item()
                total_preds += batch_labels.size(0)

                all_labels.extend(batch_labels.cpu().numpy())
                all_predictions.extend(preds.squeeze().cpu().numpy())

        accuracy = correct_preds / total_preds
        f1 = f1_score(all_labels, all_predictions)

        if use_wandb:
            wandb.log({
                "validation_loss": validation_loss / len(val_loader),
                "accuracy": accuracy,
                "f1_score": f1,
            })

        print(f"{letter}: epoch {epoch + 1}/{num_epochs}, Validation Loss: {validation_loss / len(val_loader)}, Accuracy: {accuracy}, F1 Score: {f1}")

    if use_wandb:
        # Saving the trained model as a Weights & Biases artifact
        create_and_log_artifact(model.state_dict(), f"trained_model_{letter}", "model", f"Trained model for letter {letter}")

        wandb.finish()

    return model