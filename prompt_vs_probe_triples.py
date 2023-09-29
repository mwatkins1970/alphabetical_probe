# This iterates through the list of all-Roman tokens with more than two letters.
# It attempts to deduce first three letters of spelling with two approaches
# prompt-based (iterative multishot prompting, encouraging S-P-E-L-L-I-N-G-style outputs)
# probe-based (looking for closest of the 26 "first letter" linear probes, then the closest of the 26 linear probes for pairs starting with that letter, etc.)
# Results are currently stored as a list of 6-tuples called spell_record.

# WE NEED TO FIRST EXECUTE
#setup.py
#gpt-j_setup.py
#class_setup.py
#tokens_setup.py



device = "cpu"

probe_weights_tensor = torch.load(os.path.join(GDRIVE_PATH, 'starting_letter_probe_weights_tensor.pt'))

GPTmodel = GPTmodel.to(torch.float32)

# Define a 'patience' value for early stopping:
patience = 10

# Define number of samples in training+validation dataset:
num_samples = 10000

# Define number of training epochs:
num_epochs = 100

spell_record = []

for idx in all_rom_token_gt2_indices:

  token = token_strings[idx]
  print(f"TOKEN: '{token}'")
  actual_triple = token.lstrip().lower()[0:3]

  prompt_triple = get_prompt_based_starting_triple(idx)

  probe_triple = get_probe_based_starting_triple(idx, probe_weights_tensor, GDRIVE_PATH	)


  if prompt_triple.lower() == actual_triple and probe_triple.lower() == actual_triple:
    print(f"\nBOTH PROMPT- AND PROBE-BASED SPELLINGS CORRECT\n")
  else:
    print(f"\nPROMPT-BASED: {prompt_triple}; PROBE-BASED: {probe_triple.upper()}\n")
  spell_record.append((idx, token, prompt_triple.upper(), prompt_triple.lower() == actual_triple, probe_triple.upper(), probe_triple.lower() == actual_triple))