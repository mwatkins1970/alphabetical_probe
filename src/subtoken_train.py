import torch 
import torch.nn as nn 
import torch.optim as optim
import wandb 
import numpy as np

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# These have been changed so this module runs in Colab
try:
    from src.dataset import LetterDataset
except ImportError:
    from dataset import LetterDataset

try:
    from src.probes import LinearProbe
except ImportError:
    from probes import LinearProbe

try:
    from src.MLPs import MLPProbe
except ImportError:
    from MLPs import MLPProbe

try:
    from src.get_training_data import get_training_data
except ImportError:
    from get_training_data import get_training_data


def create_and_log_artifact(tensor, name, artifact_type, description):
    # Save the tensor to a file
    filename = f"{name}.pt"
    torch.save(tensor, filename)

    # Create a new artifact
    artifact = wandb.Artifact(
        name=name,
        type=artifact_type,
        description=description,
    )
    artifact.add_file(filename)

    # Log the artifact
    wandb.log_artifact(artifact)

def train_subtoken_probe_runner(
        embeddings,
        token_strings,
        all_rom_token_indices,
        num_samples,
        num_epochs,
        device,
        probe_type,
        criteria_mode,
        use_wandb = False,
        wandb_group_name = None,
        ):

        if use_wandb:

                config = {
                    "model_name": "gpt-j",
                    "probe_type": probe_type,
                    "train_test_split": 0.2,
                    "case_sensitive": True,
                    "batch_size": 32,
                    "learning_rate": 0.001,
                    "num_samples": num_samples,
                    "num_epochs": num_epochs,
                    "device": device,
                }
                
                wandb.init(
                    project="subtoken_probe",
                    group=wandb_group_name,
                    config=config,
                )

        embeddings_dim = embeddings.shape[1]
        probe_weights_tensor = torch.zeros(2 * embeddings_dim).to(device)
        
        # construct tensors of embeddings and labels for training and validation

        all_embeddings, all_labels = get_subtoken_training_data(
        num_samples, embeddings, all_rom_token_indices, token_strings)
        
        # split the data into training and validation sets (using a function from the sklearn.model_selection module)
        X_train, X_val, y_train, y_val = train_test_split(all_embeddings, all_labels, test_size=0.2, random_state=42, stratify=all_labels)
        
        # Initialize model and optimizer based on probe_type
        if probe_type == 'linear':
                model = LinearProbe(2 * embeddings_dim).to(device)
        elif probe_type == 'mlp':
                model = MLPProbe(2 * embeddings_dim, hidden_dim=128, num_hidden_layers=2).to(device)
        
        optimizer = optim.Adam(model.parameters(), lr = 0.001)
        criterion = nn.BCEWithLogitsLoss()         # Binary cross-entropy loss with logits (because we haven't used an activation in our model)
                                # This combines sigmoid activation, which converts logits to probabilities, and binary cross entropy loss
                    # outputs will be probabilities 0 < p < 1 that the letter belongs to the token. The label will be 0 or 1 (it doesn't or it does).
        
        # create DataLoader for your training dataset
        train_dataset = LetterDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        
        #X_train, y_train (embeddings and labels for training) were created above using standard methods applied to all_embeddings and all_labels tensors
        
        # create DataLoader for your validation dataset
        val_dataset = LetterDataset(X_val, y_val)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        #X_val, y_val (embeddings and labels for validation) were likewise created above using standard methods applied to all_embeddings and all_labels tensors
        
        # TRAINING LOOP
        
        # initialise relevant variables for early stopping
        best_val_loss = float('inf')
        no_improve_count = 0
        
        print('\n_________________________________________________\n')
        for epoch in range(num_epochs):
                model.train()  # Set the model to training mode
                total_loss = 0.0

                for batch_embeddings, batch_labels in train_loader:
                        # Move your data to the chosen device during the training loop and ensure they're float32
                        # By explicitly converting to float32, you ensure that the data being fed into your model has the expected data type, and this should resolve the error you en
                        batch_embeddings = batch_embeddings.to(device).float()
                        batch_labels = batch_labels.to(device).float()  

                        optimizer.zero_grad()
                        outputs = model(batch_embeddings).squeeze()
                        loss = criterion(outputs, batch_labels)
                        loss.backward()
                        optimizer.step()  

                        total_loss += loss.item()
                        
                        if use_wandb:
                                wandb.log({"loss": loss.item()})
                
                print(f"TOKEN SUBSTRING CLASSIFICATION: epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader)}")

                # STORE THE PROBE WEIGHTS (or "direction" in embedding space associated with this probe)
                # The ord(letter) - ord('A') part is just an index from 0 to 25 corresponding to A to Z.
                probe_weights_tensor = model.fc.weight.data.clone().detach()

                # EVALUATION (VALIDATION) PHASE

                # Set the model to evaluation mode
                model.eval()

                # Create DataLoader for validation data
                val_dataset = LetterDataset(X_val, y_val)
                val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

                # Keep track of correct predictions and total predictions
                correct_preds = 0
                total_preds = 0
                validation_loss = 0.0

                with torch.no_grad():  # Ensure no gradients are computed during validation
                        all_labels = []  # Store all true labels
                        all_predictions = []  # Store all model predictions

                        for batch_embeddings, batch_labels in val_loader:
                                batch_embeddings = batch_embeddings.to(device).float()  # Ensure embeddings are on the correct device and dtype
                                batch_labels = batch_labels.to(device).float()  # Ensure labels are on the correct device and dtype

                                outputs = model(batch_embeddings).squeeze()

                                # Calculate loss on validation data
                                loss = criterion(outputs, batch_labels)
                                validation_loss += loss.item()  # Update validation loss

                                # Convert outputs to probabilities
                                probs = torch.sigmoid(outputs)
                                predictions = (probs > 0.5).float()

                                # Update correct and total predictions
                                correct_preds += (predictions == batch_labels).sum().item()
                                total_preds += batch_labels.size(0)

                                # Append batch labels and predictions to all_labels and all_predictions
                                all_labels.append(batch_labels.cpu().numpy())
                                all_predictions.append(predictions.cpu().numpy())

                        # Flatten all_labels and all_predictions lists and convert to numpy arrays
                        all_labels = np.concatenate(all_labels)
                        all_predictions = np.concatenate(all_predictions)

                        # Compute F1 Score
                        f1 = f1_score(all_labels, all_predictions)

                        validation_loss /= len(val_loader)  # Get the average validation loss

                        # Calculate accuracy and average loss
                        accuracy = correct_preds / total_preds
                        print(f"Validation Accuracy: {accuracy * 100:.2f}%")
                        print(f"Validation Loss: {validation_loss:.4f}")
                        print(f"F1 Score: {f1:.4f}\n")

                        if use_wandb:
                                wandb.log({"validation_loss": validation_loss})
                                wandb.log({"validation_accuracy": accuracy})
                                wandb.log({"f1_score": f1})

                # Before returning the tensor, log it as an artifact if wandb logging is used
                if use_wandb:
                    artifact_name = f"probe_weights_for_subtoken_detection"
                    create_and_log_artifact(
                        probe_weights_tensor,
                        artifact_name,
                        "model_tensors",
                        f"Probe weight tensor for subtoken detection task"
                    )
    
        return probe_weights_tensor
                    
        # print(probe_weights_tensor)
