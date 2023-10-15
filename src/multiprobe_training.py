# This was created to train multilinear probes

import torch 
import torch.nn as nn 
import torch.optim as optim
import wandb 
import numpy as np
import tempfile
import os

from multiprobes import multi_dim_loss
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# These have been changed so this module runs in Colab
try:
    from src.dataset import LetterDataset
except ImportError:
    from dataset import LetterDataset

try:
    from src.multiprobes import MultiDimensionalProbe
except ImportError:
    from multiprobes import MultiDimensionalProbe

try:
    from src.get_training_data import get_training_data
except ImportError:
    from get_training_data import get_training_data


# random seeding
rnd_seed = 42

torch.manual_seed(rnd_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(rnd_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

np.random.seed(rnd_seed)

hidden_dim=4096
num_hidden_layers=2

def create_and_log_artifact(tensor, name, artifact_type, description):
    # Use a temporary file to save the tensor
    with tempfile.NamedTemporaryFile(delete=True, suffix='.pt') as tmp:
        torch.save(tensor, tmp.name)
        
        # Create a new artifact
        artifact = wandb.Artifact(
            name=name,
            type=artifact_type,
            description=description,
        )
        artifact.add_file(tmp.name)
        
        # Log the artifact
        wandb.log_artifact(artifact)

def all_multiprobe_training_runner(
        embeddings, 
        all_rom_token_indices, 
        token_strings,
        alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ",   
        num_samples = 10000, # Define number of samples in training+validation dataset:
        num_epochs = 5, # Define number of training epochs:
        batch_size = 128,
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        use_wandb = False,
        probe_type = 'linear',
        criteria_mode = "anywhere",   # "anywhere", "starting" or "posN" (where N is a digit)
        ):

        if use_wandb:
                # generate unique run name
                group_name = wandb.util.generate_id() + "_" + alphabet

        # Initialize an empty tensor to store the learned weights for all letters (or, equivalently, 26 "directions", one for each linear probe)
        embeddings_dim = embeddings.shape[1]
        all_probe_weights = {letter: None for letter in alphabet}

        # Now loop over the alphabet and train/validate a probe for each letter:

        for i, letter in enumerate(alphabet):

                # Train the probe for the current letter:
                all_probe_weights[letter] = train_letter_probe_runner(
                letter,   
                embeddings,
                token_strings,
                all_rom_token_indices,
                num_samples,
                num_epochs,
                batch_size,
                device,
                probe_type,
                criteria_mode,
                use_wandb,
                wandb_group_name = group_name if use_wandb else None
                )

        if use_wandb:
            # Start a new W&B run for logging the aggregate artifact
            wandb.init(project="letter_presence_probes", name="aggregate_artifact_logging")
            create_and_log_artifact(all_probe_weights, "all_probe_weights", "model_tensors", "All case-insensitive letter presence probe weights tensor")
            wandb.finish()  # End the run

def train_letter_probe_runner(
        letter,      
        embeddings,
        token_strings,
        all_rom_token_indices,
        num_samples,
        num_epochs,
        batch_size,
        device,
        probe_type,
        criteria_mode,
        use_wandb = False,
        wandb_group_name = None,
        ):

        if use_wandb:

                config = {
                    "letter": letter,            
                    "criteria_mode": criteria_mode,
                    "model_name": "gpt-j",
                    "probe_type": probe_type,
                    "train_test_split": 0.2,
                    "seed": rnd_seed,
                    "case_sensitive": False,
                    "batch_size": 128,
                    "learning_rate": 0.001,
                    "num_samples": num_samples,
                    "num_epochs": num_epochs,
                    "device": device,
                }
                
                wandb.init(
                    project="letter_presence_probes",
                    group=wandb_group_name,
                    config=config,
                )

        embeddings_dim = embeddings.shape[1]
        probe_weights_tensor = torch.zeros(embeddings_dim).to(device)
        
        # construct tensors of embeddings and labels for training and validation

        all_embeddings, all_labels = get_training_data(
        criteria_mode, letter, num_samples, embeddings, all_rom_token_indices, token_strings)
        
        # split the data into training and validation sets (using a function from the sklearn.model_selection module)
        X_train, X_val, y_train, y_val = train_test_split(all_embeddings, all_labels, test_size=0.2, random_state=rnd_seed, stratify=all_labels)
        
        # Initialize model and optimizer
        model = LinearProbe(embeddings_dim).to(device)
        
        optimizer = optim.Adam(model.parameters(), lr = 0.001)
        criterion = nn.BCEWithLogitsLoss()         # Binary cross-entropy loss with logits (because we haven't used an activation in our model)
                                # This combines sigmoid activation, which converts logits to probabilities, and binary cross entropy loss
                    # outputs will be probabilities 0 < p < 1 that the letter belongs to the token. The label will be 0 or 1 (it doesn't or it does).
        
        # create DataLoader for your training dataset
        train_dataset = LetterDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        #X_train, y_train (embeddings and labels for training) were created above using standard methods applied to all_embeddings and all_labels tensors            
        #X_val, y_val (embeddings and labels for validation) were likewise created above using standard methods applied to all_embeddings and all_labels tensors
        
        # TRAINING LOOP
        
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

                  outputs = model(batch_embeddings)
                  loss = multi_dim_loss(outputs, batch_labels)  # use the new loss function

                  loss.backward()
                  optimizer.step()  

                  total_loss += loss.item()
                        
                  if use_wandb:

                      wandb.log({"loss": loss.item()})
                
            print(f"{letter}: epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader)}")


                
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

                    if probe_type == 'linear':
                        probe_weights_tensor = model.weight.data.clone().detach()  
                        if use_wandb:
                            artifact_name = f"probe_weights_for_{letter.upper()}_criterion_{criteria_mode}"
                            print("Now logging probe weight tensor wandb artifact...")
                            create_and_log_artifact(
                                probe_weights_tensor,
                                artifact_name,
                                "model_tensors",
                                f"Letter presence probe weight tensor for {letter.upper()} with criterion_{criteria_mode}"
                            )

                    elif probe_type == 'mlp':
                        # Skip logging weights for mlp, since you're not storing or using them
                        pass


                    # Calculate accuracy and average loss
                    accuracy = correct_preds / total_preds
                    print(f"Validation Accuracy: {accuracy * 100:.2f}%")
                    print(f"Validation Loss: {validation_loss:.4f}")
                    print(f"F1 Score: {f1:.4f}\n")

                    if use_wandb:
                        wandb.log({
                            "validation_loss": validation_loss,
                            "validation_accuracy": accuracy,
                            "f1_score": f1
                        })

        if use_wandb:
            wandb.finish()

        return probe_weights_tensor