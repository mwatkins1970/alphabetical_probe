# This function records results for probe- and prompt-based attempts to predict the first letter of all-roman tokens
# 1-shot seems to work best
# returns results as both a python dictionary and as a pandas dataframe

import torch
import wandb
import pandas as pd
from find_closest_probe import find_closest_probe
device = "cpu"

def first_letter_evals_runner(GPTmodel, tokenizer, embeddings, token_strings, all_rom_token_gt2_indices, index_list, num_shots):

    use_wandb = False

    if use_wandb:
          wandb.init(project="SpellingMiracleCollab", name="first letter prompt vs probe evals")

    # Currently loading in pre-calculated shape(26,4096) tensor of all 26 first-letter probes
    # Replace with commented out line if you want to start by trained these first.
    # probe_weights_tensor = all_probe_training_runner(embeddings, all_rom_token_indices, token_strings, probe_type = 'linear', use_wandb = True, criteria_mode = "pos1")
    probe_weights_tensor = torch.load('/content/Drive/My Drive/SpellingMiracleCollab/pos1_probe_weights_tensor.pt')

    results_dict = {}
    prompt_correct_count = 0
    prompt_wrong_count = 0
    probe_correct_count = 0
    probe_wrong_count = 0

    preprompts = []

    preprompts.append('''The string "''')
    preprompts.append('''The string " heaven" begins with the letter "H".\nThe string "''')
    preprompts.append('''The string " heaven" begins with the letter "H".\nThe string "same" begins with the letter "S".\nThe string "''')
    preprompts.append('''The string " heaven" begins with the letter "H".\nThe string "same" begins with the letter "S".\nThe string " altitude" begins with the letter "A".\nThe string "''')
    preprompts.append('''The string " heaven" begins with the letter "H".\nThe string "same" begins with the letter "S".\nThe string " altitude" begins with the letter "A".\nThe string "Trump" begins with the letter "T".\nThe string "''')
    
    results_dict["number of shots"] = num_shots
    results_dict["prompt template"] = preprompts[num_shots] + '''<token>" begins with the letter "'''
    results_dict["predictions"] = []


    for i in index_list:

        token_index = token_strings.index(token_strings[all_rom_token_gt2_indices[i]])

        token = token_strings[token_index]

        emb = embeddings[token_index]

        closest_probe, probe_distance_list = find_closest_probe(emb, probe_weights_tensor)

        prompt = preprompts[num_shots] + token + '''" begins with the letter "'''

        print(f"INDEX: {token_index};  TOKEN: '{token}'\nPROMPT:\n{prompt}")

        ix = tokenizer.encode(prompt)

        model_out = GPTmodel.generate(
                torch.tensor(ix).unsqueeze(0).to(device),
                max_length=len(ix) + 1,
                temperature=0.00000000001,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
        )

        output = tokenizer.decode(model_out[0])[len(prompt):]

        probe_correct = (closest_probe.lower() == token.lstrip().lower()[0])

        if probe_correct:
          probe_correct_count += 1
        else:
          probe_wrong_count +=1

        prompt_correct = (output.lower() == token.lstrip().lower()[0])
           
        if prompt_correct:
          prompt_correct_count += 1
        else:
          prompt_wrong_count +=1

        
        single_token_results_dict = {}
        single_token_results_dict["index"] = token_index
        single_token_results_dict["token"] = token
        single_token_results_dict["first letter"] = token.lstrip()[0]
        single_token_results_dict["prompt prediction"] = output.upper()
        single_token_results_dict["probe prediction"] =  closest_probe.upper()
        single_token_results_dict["probe cos similarities"] =  probe_distance_list

        results_dict["predictions"].append(single_token_results_dict)
  
        print(f"\nPROMPT PREDICTION: {output} ({prompt_correct})")
        print(f"PROBE PREDICTION: {closest_probe.upper()} ({probe_correct})")
        print(f"Current prompt-based prediction score: {prompt_correct_count}/{prompt_correct_count + prompt_wrong_count} ({100*prompt_correct_count/(prompt_correct_count + prompt_wrong_count):.2f}%)")
        print(f"Current probe-based prediction score: {probe_correct_count}/{probe_correct_count + probe_wrong_count} ({100*probe_correct_count/(probe_correct_count + probe_wrong_count):.2f}%)")
        print("-"*50)

        print('\n')

    if use_wandb:
        wandb.log({"results": results_dict})

    df = pd.DataFrame(results_dict["predictions"])
    df["number of shots"] = results_dict["number of shots"]
    df["prompt template"] = results_dict["prompt template"]

    return df, results_dict
