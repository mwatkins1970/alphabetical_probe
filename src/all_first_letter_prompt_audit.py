# This runs just the prompt 'first letter' evals for the entire set of all-Roman tokens
 
# No mutation is involved with the token embeddings.

# This outputs a dataframe and a dictionary with prompt and probe predictions, all first-letter probe distances and metadata

import random
import torch

def all_first_letter_evals(GPTmodel, tokenizer, embeddings, token_strings, all_rom_token_gt2_indices, index_list, num_shots):

    device = "cpu"
    
    results_dict = {}
    prompt_correct_count = 0
    prompt_wrong_count = 0

    preprompts = []

    preprompts.append('''The string "''')
    preprompts.append('''The string " heaven" begins with the letter "H".\nThe string "''')
    preprompts.append('''The string " heaven" begins with the letter "H".\nThe string "same" begins with the letter "S".\nThe string "''')
    preprompts.append('''The string " heaven" begins with the letter "H".\nThe string "same" begins with the letter "S".\nThe string " altitude" begins with the letter "''')
    preprompts.append('''The string " heaven" begins with the letter "H".\nThe string "same" begins with the letter "S".\nThe string " altitude" begins with the letter "A".\nThe string "Trump" begins with the letter "T".\nThe string "''')
    
    results_dict["number of shots"] = num_shots
    results_dict["prompt template"] = preprompts[num_shots] + '''<token>" begins with the letter "'''
    results_dict["predictions"] = []

    for token_index in index_list:

        token = token_strings[token_index]

        emb = embeddings[token_index]

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

        results_dict["predictions"].append(single_token_results_dict)
  
        print(f"\nPROMPT PREDICTION: {output} ({prompt_correct})")
        print(f"Current prompt-based prediction score: {prompt_correct_count}/{prompt_correct_count + prompt_wrong_count} ({100*prompt_correct_count/(prompt_correct_count + prompt_wrong_count):.2f}%)")
        print("-"*50)

        print('\n')

def all_first_letter_prompt_audit(GPTmodel, tokenizer, embeddings, all_rom_token_gt2_indices, token_strings, num_shots):

    # The range is the list all_rom_token_gt2_indices (don't confuse with actual token indices)
    index_list = all_rom_token_gt2_indices

    df, results = all_first_letter_evals(GPTmodel, tokenizer, embeddings, token_strings, all_rom_token_gt2_indices, index_list, num_shots)

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
