#THIS RECORDS RESULTS FOR GPT-Js ABILITY TO INFER THE FIRST LETTER OF THE VAST MAJORITY OF ALL-ROMAN TOKENS
#USING BOTH PROMPTING AND PROBE DISTANCES

def first_letter_evals_runner(GPTmodel, tokenizer, embeddings, token_strings, all_rom_token_gt2_indices, range_start, range_end, num_shots):

    import torch
    import wandb
    from find_closest_probe import find_closest_probe
    device = "cpu"

    use_wandb = False

    if use_wandb:
          wandb.init(project="SpellingMiracleCollab", name="first letter prompt vs probe evals")

    # probe_weights_tensor = all_probe_training_runner(embeddings, all_rom_token_indices, token_strings, probe_type = 'linear', use_wandb = True, criteria_mode = "pos1")
    probe_weights_tensor = torch.load('/content/Drive/My Drive/SpellingMiracleCollab/pos1_probe_weights_tensor.pt')

    results_dict = {}
    prompt_correct = 0
    prompt_wrong = 0
    probe_correct = 0
    probe_wrong = 0

    preprompts = []

    preprompts.append('''The string "''')
    preprompts.append('''The string " heaven" begins with the letter "H".\nThe string "''')
    preprompts.append('''The string " heaven" begins with the letter "H".\nThe string "same" begins with the letter "S".\nThe string "''')
    preprompts.append('''The string " heaven" begins with the letter "H".\nThe string "same" begins with the letter "S".\nThe string " altitude" begins with the letter "''')
    preprompts.append('''The string " heaven" begins with the letter "H".\nThe string "same" begins with the letter "S".\nThe string " altitude" begins with the letter "A".\nThe string "Trump" begins with the letter "T".\nThe string "''')
    
    results_dict["number of shots"] = num_shots
    results_dict["prompt template"] = preprompts[num_shots] + '''<token>" begins with the letter "'''
    results_dict["intervention type"] = None
    results_dict["intervention scale"] = None
    results_dict["predictions"] = []


    for idx in all_rom_token_gt2_indices[range_start:range_end]:

        token = token_strings[idx]

        emb = embeddings[idx]

        closest_probe, probe_distance_list = find_closest_probe(emb, probe_weights_tensor)

        prompt = preprompts[num_shots] + token + '''" begins with the letter "'''

        print(f"PROMPT:\n{prompt}")

        ix = tokenizer.encode(prompt)

        model_out = GPTmodel.generate(
                torch.tensor(ix).unsqueeze(0).to(device),
                max_length=len(ix) + 1,
                temperature=0.00000000001,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
        )

        output = tokenizer.decode(model_out[0])[len(prompt):]    
        if output.lower() == token.lstrip().lower()[0]:
          prompt_correct += 1
        else:
          prompt_wrong +=1

        if closest_probe.lower() == token.lstrip().lower()[0]:
          probe_correct += 1
        else:
          probe_wrong +=1
        
        single_token_results_dict = {}
        single_token_results_dict["index"] = idx
        single_token_results_dict["token"] = token
        single_token_results_dict["first letter"] = token.lstrip()[0]
        single_token_results_dict["prompt prediction"] = output
        single_token_results_dict["probe prediction"] =  closest_probe
        single_token_results_dict["probe cos similarities"] =  probe_distance_list

        results_dict["predictions"].append(single_token_results_dict)

        print(f"PROMPT PREDICTION: {output}")
        print(f"PROMPT CORRECT: {prompt_correct}/{prompt_correct + prompt_wrong} ({100*prompt_correct/(prompt_correct + prompt_wrong):.2f}%)")
        print(f"PROBE PREDICTION: {closest_probe.upper()}")
        print(f"PROBE CORRECT: {probe_correct}/{probe_correct + probe_wrong} ({100*probe_correct/(probe_correct + probe_wrong):.2f}%)")

        print('\n')

    if use_wandb:
        wandb.log({"results": results_dict})

    return results_dict