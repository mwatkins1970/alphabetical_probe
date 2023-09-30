# THIS FUNCTION TRAINS A PROBE TO CLASSIFY TOKENS ACCORDING TO FIRST THREE LETTERS = target_triple

def starting_triple_probe_train(target_triple):  

  #print(f"Starting triple: {target_triple}")

  # construct tensors of embeddings and labels for training and validation
  all_embeddings, all_labels = get_training_data_starting_triple(target_triple, num_samples, embeddings, all_rom_token_gt2_indices)

  if sum([1 for idx in all_rom_token_gt2_indices if token_strings[idx].lstrip()[0:3].lower() == target_triple]) > 15:   
    # arbitrary number of tokens with correct starting letters to act as minimum cutoff - PROBABLY FAR TOO LOW?

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

    print(f"{target_triple}: epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader)}")

    # STORE THE PROBE WEIGHTS (or "direction" in embedding space associated with this probe)

    letter_triple_probe_weights = model.fc.weight.data.clone().detach()


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
          torch.save(model.state_dict(), f"model_{target_triple}.pth")
          no_improve_count = 0  # Reset counter
        else:
          no_improve_count += 1

        if no_improve_count >= patience:
          break

    # Calculate accuracy and average loss
    accuracy = correct_preds / total_preds
    average_loss = validation_loss / len(val_loader)
    print(f"Validation Accuracy: {accuracy * 100:.2f}%")
    print(f"Validation Loss: {average_loss:.4f}")

  else:   # if there's not enough data, write a row of zeros for this index
    letter_triple_probe_weights = torch.zeros(4096)
    #print(f"Not enough tokens starting '{target_triple}', so writing zeros to this row of the (26,4096) tensor for the starting pair {target_triple[0:2]}.")

  return letter_triple_probe_weights