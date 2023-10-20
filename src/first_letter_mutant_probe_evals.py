import torch

from find_closest_probe import find_closest_probe
from probe_subtractor import probe_subtractor 

device = "cpu"

def first_letter_mutant_probe_evals_runner(GPTmodel, tokenizer, embeddings, token_strings, all_rom_token_gt2_indices, token_index, coeff):

    use_wandb = False

    if use_wandb:
          wandb.init(project="SpellingMiracleCollab", name="first letter prompt vs probe evals")

    # Currently loading in pre-calculated shape(26,4096) tensor of all 26 first-letter probes
    # Replace with commented out line if you want to start by trained these first.
    # probe_weights_tensor = all_probe_training_runner(embeddings, all_rom_token_indices, token_strings, probe_type = 'linear', use_wandb = True, criteria_mode = "pos1")
    probe_weights_tensor = torch.load('/content/Drive/My Drive/SpellingMiracleCollab/pos1_probe_weights_tensor.pt')

    token = token_strings[token_index]

    emb = embeddings[token_index]

    letter = token.lstrip().lower()[0]  # First letter of token string, dictates which probe we use

    emb = probe_subtractor(coeff, emb, letter)   # mutate the embedding by projecting back along probe direction, appropriately scaled

    closest_probe, probe_distance_list = find_closest_probe(emb, probe_weights_tensor)

    print(f"MUTATION COEFFICIENT: {coeff}")
    print(f"We're projecting the embedding for this token back along the '{letter.upper()}' probe direction, scaled by {coeff}")
    print(f"Nearest first-letter probe to resulting 'mutant embedding': {closest_probe.upper()}\n")
    #print(f"Full list of probe distances to this mutant embedding: {probe_distance_list}")

    return closest_probe, probe_distance_list

