# THIS TRAINS PROBES ON SETS OF THEMED WORDS

# The following function randomly samples from appropriate list to create a balanced dataset (or as balanced as possible based on available data).
# The resulting 'all_embeddings' (shape-[num_samples, 4096] tensor) and 'all_labels' (shape-[num_samples] tensor)
# are then suitable for separating into training and validation subsets.

def themed_training_data(token_idxs, num_samples, embeddings, all_rom_token_indices):

    import random 
    import torch

    # Fetch indices for tokens that fit the theme
    positive_indices = token_idxs

    # Fetch indices for tokens that do not contain the specified letter
    # (by taking a set difference and then converting back to a list)
    negative_indices = list(set(all_rom_token_indices) - set(positive_indices))

    # Randomly sample from positive and negative indices to balance the dataset
    num_positive = min(num_samples // 2, len(positive_indices))
    num_negative = num_samples - num_positive

    sampled_positive_indices = random.sample(positive_indices, num_positive)
    sampled_negative_indices = random.sample(negative_indices, num_negative)

    # Combine sampled indices
    sampled_indices = sampled_positive_indices + sampled_negative_indices
    random.shuffle(sampled_indices)  # Shuffle combined indices for randomness in training

    # Convert sampled_indices to a torch tensor, if it's not already
    sampled_indices_tensor = torch.tensor(sampled_indices, dtype=torch.long) if isinstance(sampled_indices, list) else sampled_indices

    # Ensure that embeddings is treated as a tensor, in case it's a Parameter
    embeddings_tensor = embeddings.data

    # Extract corresponding embeddings and labels
    all_embeddings = embeddings_tensor[sampled_indices_tensor]
    all_labels = torch.tensor([1 if idx in positive_indices else 0 for idx in sampled_indices_tensor])

    return all_embeddings.clone().detach(), torch.tensor(all_labels).clone().detach()  #This probably overcomplicated, but following a PyTorch warning about boosting efficiency (as advised by GPT4)


def theme_probe_trainer(themestring, token_idxs, embeddings, all_rom_token_indices):

    import torch
    import torch.nn as nn 
    import torch.optim as optim

    from torch.utils.data import DataLoader
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import f1_score

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

    # Use CUDA if possible:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Initialize an empty tensor to store the learned weights for all letters (or, equivalently, 26 "directions", one for each linear probe)
    themed_probe_weights_tensor = torch.zeros(4096)

    # Define a 'patience' value for early stopping:
    patience = 10

    # Define number of samples in training+validation dataset:
    num_samples = 10000

    # Define number of training epochs:
    num_epochs = 100

    # construct tensors of embeddings and labels for training and validation
    all_embeddings, all_labels = themed_training_data(token_idxs, num_samples, embeddings, all_rom_token_indices)

    # split the data into training and validation sets (using a function from the sklearn.model_selection module)
    X_train, X_val, y_train, y_val = train_test_split(all_embeddings, all_labels, test_size=0.2, random_state=42, stratify=all_labels)

    # Initialize model and optimizer
    model = LinearProbe(4096).to(device)
    optimizer = optim.Adam(model.parameters(), lr = 0.001)
    criterion = nn.BCEWithLogitsLoss()         # Binary cross-entropy loss with logits (because we haven't used an activation in our model)
                                  # This combines sigmoid activation, which converts logits to probabilities, and binary cross entropy loss
                    # outputs will be probabilities 0 < p < 1 that the letter belongs to the token The label will be 0 or 1 (it doesn't or it does)

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

        print(f"{themestring}: epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader)}")

    # STORE THE PROBE WEIGHTS (or "direction" in embedding space associated with this probe)
    themed_probe_weights_tensor = model.fc.weight.data.squeeze().clone().detach()


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
        for batch_embeddings, batch_labels in val_loader:
            batch_embeddings = batch_embeddings.to(device).float()  # Ensure embeddings are on the correct device and dtype
            batch_labels = batch_labels.to(device).float()  # Ensure labels are on the correct device and dtype

            outputs = model(batch_embeddings).squeeze()
            loss = criterion(outputs, batch_labels)
            validation_loss += loss.item()

            # Convert outputs to probabilities
            probs = torch.sigmoid(outputs)
            predictions = (probs > 0.5).float()

            # Update correct and total predictions
            correct_preds += (predictions == batch_labels).sum().item()
            total_preds += batch_labels.size(0)

            # Early stopping and model checkpointing
            if validation_loss < best_val_loss:
                best_val_loss = validation_loss
                best_train_loss = total_loss / len(train_loader)  # Store best training loss
                #torch.save(model.state_dict(), f"model_{themestring}.pth")
                no_improve_count = 0  # Reset counter
            else:
                no_improve_count += 1

            if no_improve_count >= patience:
                break

    # Calculate accuracy and average loss
    accuracy = correct_preds / total_preds
    average_loss = validation_loss / len(val_loader)
    validation_accuracy = accuracy * 100
    print(f"Validation Accuracy: {validation_accuracy:.2f}%")
    print(f"Validation Loss: {average_loss:.4f}")

    return themed_probe_weights_tensor
