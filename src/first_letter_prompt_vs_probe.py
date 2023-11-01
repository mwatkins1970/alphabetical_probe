# This runs the prompt and probe 'first letter' evals for a set of all-Roman tokens
# (based on index_list, where the indices are 0...44634 from all_rom_token_gt2_indices list)
 # No mutation is involved with the token embeddings.
# This outputs a dataframe and a dictionary with prompt and probe predictions, all first-letter probe distances and metadata

import random
from first_letter_evals import first_letter_evals_runner


def first_letter_prompt_vs_probe(GPTmodel, tokenizer, embeddings, all_rom_token_gt2_indices, token_strings, num_shots, k):

    num_indices = len(all_rom_token_gt2_indices)    # 44634

    selected_indices = random.sample(range(num_indices), k)
    selected_indices.sort()

    # The range is the list all_rom_token_gt2_indices (don't confuse with actual token indices)
    index_list = selected_indices

    df, results = first_letter_evals_runner(GPTmodel, tokenizer, embeddings, token_strings, all_rom_token_gt2_indices, index_list, num_shots)

    # results is a dictionary which inclues all the prompt and probe predictions

    count = 0

    for prediction in results["predictions"]:

        first_letter = prediction['first letter'].lstrip().lower()    # Actual (not predicted) first letter
        prompt_prediction = prediction['prompt prediction'].lstrip().lower()   # Prompt-based prediction of first letter

        if first_letter == prompt_prediction:
            count += 1

    print(f"That's prompt-based success on {count}/{(len(index_list))} tokens")
    print(f"'results' DICTIONARY: {results}")
    print('\n')

    return results
