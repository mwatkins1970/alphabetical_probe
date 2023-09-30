def train_num_distinct_letters_probes(embeddings, distinct_letters_dict, all_rom_token_indices): 
  
  # Use CUDA if possible:
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  # Initialize an empty tensor to store the learned weights for all letters (or, equivalently, 26 "directions", one for each linear probe)
  distinct_letters_probe_weights_tensor = torch.zeros(26, 4096)

  # Define a 'patience' value for early stopping:
  patience = 10

  # Define number of samples in training+validation dataset:
  num_samples = 10000

  # Define number of training epochs:
  num_epochs = 100


  # Now loop over the alphabet and train/validate a linear probe for each letter:

  for num_distinct_letters in range(1,13):    # there are not enough tokens with 13 or 14 distinct letters to work with

    # construct tensors of embeddings and labels for training and validation
    all_embeddings, all_labels = get_training_data_num_distinct_letters(num_distinct_letters, num_samples, embeddings, distinct_letters_dict, all_rom_token_indices)

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

        print(f"{num_letters} distinct letters: epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader)}")

    # STORE THE PROBE WEIGHTS (or "direction" in embedding space associated with this probe)
    distinct_letters_probe_weights_tensor[num_distinct_letters] = model.fc.weight.data.clone().detach()


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
                torch.save(model.state_dict(), f"model_{num_distinct_letters}.pth")
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


  # OUTPUT SUMMARY 

  print("\nSummary:")
  print("No. distinct letters | Best Train Loss | Validation Loss | Validation Accuracy")
  print("-" * 75)
  for num_letters, metrics in results.items():
      print(f"{num_distinct_letters}                | {metrics['best_train_loss']:.4f}           | {metrics['validation_loss']:.4f}        | {metrics['validation_accuracy']:.4f}")

  print("-" * 75)

  print(distinct_letters_probe_weights_tensor)

  return distinct_letters_probe_weights_tensor