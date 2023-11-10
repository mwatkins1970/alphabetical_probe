import os
import requests
import torch

from find_closest_probe import find_closest_probe
from probe_subtractor import probe_subtractor 

def load_probe_weights_tensor(filename='pos1_probe_weights_tensor.pt'):
    # Check if the file exists locally; if not, download it
    if not os.path.isfile(filename):
        url = 'https://github.com/mwatkins1970/alphabetical_probe/raw/main/pos1_probe_weights_tensor.pt'
        r = requests.get(url)
        with open(filename, 'wb') as f:
            f.write(r.content)

    # Load the tensor using PyTorch
    probe_weights_tensor = torch.load(filename)
    return probe_weights_tensor


device = "cpu"

def first_letter_mutant_probe_evals_runner(GPTmodel, tokenizer, embeddings, token_strings, all_rom_token_gt2_indices, token_index, coeff):

    use_wandb = False

    if use_wandb:
          wandb.init(project="SpellingMiracleCollab", name="first letter prompt vs probe evals")

    probe_weights_tensor = load_probe_weights_tensor()

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
