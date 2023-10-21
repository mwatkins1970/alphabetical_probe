# We're now going to extend the 'results' dictionary to 'extended_results' which will include key/value pairs relating to
# prompting with mutated tokens (with a range of coeff values)

# Note that 'coeff' controls "how far back" we linearly project int he prove direction - 2 is default for reflection
# But greater than 2 corresponds to making the probe *even less likely* to classify the mutant embedding positively

import os
import pickle

from mutant_first_letter_evals import mutant_first_letter_evals_runner
from probe_subtractor import probe_subtractor
from find_closest_probe import find_closest_probe

def mutation_evals_runner(results, upper_range, step, GPTmodel, tokenizer, embeddings, token_strings, all_rom_token_gt2_indices, num_shots):

    coeffs = [0, 1] + list(range(2, upper_range + 1, step))  # This merges the two lists into one

    save_path = "/content/Drive/My Drive/SpellingMiracleCollab/first_letter_extended_results_checkpoints.pkl"  
    
    if os.path.exists(save_path):
        # If a save file exists, load the progress from it.
        with open(save_path, "rb") as f:
            extended_results = pickle.load(f)
    else:
        extended_results = results  # Make a copy of results dictionary if no save file exists.

    predictions = extended_results['predictions']  # We're going to enrich this and then replace the version in the copied dictionary.

    for count, prediction_dict in enumerate(predictions):  # prediction_dict is a dictionary for a single token, we'll add to it
        if 'mutation predictions' not in prediction_dict:
            prediction_dict['mutation predictions'] = {}  # initiate a dictionary for the prompt-based predictions when embedding is mutated
            index = prediction_dict['index']

            print(f"TOKEN: '{prediction_dict['token']}'")

            switch_flag = False

            for coeff in coeffs:
                print(f"MUTATION COEFFICIENT: {coeff}")
                _, results_mutated, switch_flag = mutant_first_letter_evals_runner(GPTmodel, tokenizer, embeddings, token_strings, all_rom_token_gt2_indices, index, num_shots, coeff, switch_flag)
                # make the value for the coeff-key an ordered pair - prediction and logit)
                prediction_dict['mutation predictions'][coeff] = (results_mutated['predictions'][0]['prompt prediction'], results_mutated['predictions'][0]['prediction logit'])

            print(f"{count + 1}/{len(predictions)}: Results dictionary now enriched to include first-letter predictions for mutations of embedding of token '{prediction_dict['token']}'\n")

            # Save the progress after processing each token
            with open(save_path, "wb") as f:
                pickle.dump(extended_results, f)

    return extended_results
