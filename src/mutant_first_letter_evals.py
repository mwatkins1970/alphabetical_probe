# This function records results for probe- and prompt-based attempts to predict the first letter of all-roman tokens
# 1-shot seems to work best
# returns results as both a python dictionary and as a pandas dataframe

import torch
import wandb
from find_closest_probe import find_closest_probe
from probe_subtractor import probe_subtractor_alt
device = "cpu"

def mutant_first_letter_evals_runner(embeddings, token_strings, token_list, coeff):

    use_wandb = False

    if use_wandb:
          wandb.init(project="SpellingMiracleCollab", name="first letter prompt vs probe evals")

    # Currently loading in pre-calculated shape(26,4096) tensor of all 26 first-letter probes
    # Replace with commented out line if you want to start by trained these first.
    # probe_weights_tensor = all_probe_training_runner(embeddings, all_rom_token_indices, token_strings, probe_type = 'linear', use_wandb = True, criteria_mode = "pos1")
    probe_weights_tensor = torch.load('/content/Drive/My Drive/SpellingMiracleCollab/pos1_probe_weights_tensor.pt')

    results_dict = {}

    results_dict["intervention type"] = "orthogonal pushback"
    results_dict["intervention scale"] = coeff
    results_dict["predictions"] = []


    for token in token_list:

        idx = token_strings.index(token)

        emb = embeddings[idx]

        emb_mut = probe_subtractor_alt(coeff, emb, token.lstrip().lower()[0])

        closest_probe, probe_distance_list = find_closest_probe(emb_mut, probe_weights_tensor)

        single_token_results_dict = {}
        single_token_results_dict["index"] = idx
        single_token_results_dict["token"] = token
        single_token_results_dict["first letter"] = token.lstrip()[0]
        single_token_results_dict["probe prediction"] =  closest_probe
        single_token_results_dict["probe cos similarities"] =  probe_distance_list

        results_dict["predictions"].append(single_token_results_dict)

        print(f"PROBE PREDICTION: {closest_probe.upper()}")

        print('\n')

    if use_wandb:
        wandb.log({"results": results_dict})

    return results_dict
