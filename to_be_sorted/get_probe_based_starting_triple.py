def get_probe_based_starting_triple(idx, weights_tensor, GDRIVE_PATH):

  print("Probe-based prediction:")

  # FIRST LETTER: LOOK FOR NEAREST FIRST-LETTER PROBE IN THE 26-ROW TENSOR 
  pred_first_letter = find_closest_probe(embeddings[idx], weights_tensor)
  print(f"PREDICTED FIRST LETTER: {pred_first_letter.upper()}")
      
  
  # SECOND LETTER: LOOK FOR THE NEAREST FIRST-LETTER PROBE IN THE APPROPRIATE 26-ROW TENSOR (GIVEN FIRST LETTER)

  relevant_pair_probe_tensor = torch.load(os.path.join(GDRIVE_PATH, 'pairs_starting_' + pred_first_letter + '_tensor.pt'))
	
  pred_second_letter = find_closest_probe(embeddings[idx], relevant_pair_probe_tensor)
  print(f"PREDICTED SECOND LETTER: {pred_second_letter.upper()}")

  # THIRD LETTER: IF IT HASN'T ALREADY BEEN BULIT AND SAVED, WE NOW NEED TO TRAIN 26 THIRD-LETTER PROBES (FOR THIS STARTING PAIR)
  # AND BUNDLE THEM INTO A 26-ROW TENSOR

  if os.path.exists(os.path.join(GDRIVE_PATH,'triples_starting_' + pred_first_letter + pred_second_letter + '_tensor.pt')):
      triple_probe_weights_tensor = torch.load(os.path.join(GDRIVE_PATH,'triples_starting_' + pred_first_letter + pred_second_letter + '_tensor.pt'))
  else:
      print("\nSorry, I just have to train up to 26 linear probes for the relevant three letter combinations (and then pick the closest one).")
      print("Might take a minute.")
      triple_probe_weights_tensor = torch.zeros(26,4096)

      for k, char in enumerate('abcdefghijklmnopqrstuvwxyz'):

        target_start_triple = pred_first_letter + pred_second_letter + char

        triple_probe_weights_tensor[k] = starting_triple_probe_train(target_start_triple)
        # SO THIS SHOULD SEQUENTIALLY WRITE IN THE 26 PROBE VECTOR/DIRECTIONS (A ROW OF ZEROS WHERE TRAINING ISN'T POSS.)

      torch.save(triple_probe_weights_tensor, os.path.join(GDRIVE_PATH,'triples_starting_' + pred_first_letter + pred_second_letter + '_tensor.pt'))
      #SAVE TO AVOID DUPLICATING EFFORT 

  pred_third_letter = find_closest_probe(embeddings[idx], triple_probe_weights_tensor)
  print(f"PREDICTED THIRD LETTER: {pred_third_letter.upper()}")

  return pred_first_letter + pred_second_letter + pred_third_letter