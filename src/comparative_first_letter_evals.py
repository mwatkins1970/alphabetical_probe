import torch
import wandb
import pandas as pd
from find_closest_probe import find_closest_probe
from probe_subtractor import probe_subtractor
from transformers import GPTJForCausalLM

device = "cpu"

temperature = 0.00000000001

def comparative_first_letter_evals_runner(GPTmodel, tokenizer, embeddings, token_strings, all_rom_token_gt2_indices, index_list, num_shots, coeff):

    use_wandb = False

    if use_wandb:
          wandb.init(project="SpellingMiracleCollab", name="first letter prompt vs probe evals")

    results_dict = {}

    failure_tokens = []

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

    pred_correct = 0
    pred_wrong = 0

    # iterate through our list of token indices (each token will have its embedding mutated before being inserted into the prompt template and forward-passed)

    for idx in index_list:   
        token = token_strings[idx]
        emb = embeddings[idx]

        prompt = preprompts[num_shots] + token + '''" begins with the letter "'''
        print(f"PROMPT:\n{prompt}")

        encoded_prompt = tokenizer.encode(prompt, return_tensors="pt")
        prompt_length = encoded_prompt.size(1)

        # Prepare the embeddings
        with torch.no_grad():
            prompt_emb_tensor = GPTmodel.transformer.wte(encoded_prompt)[0]  # Shape: [sequence_length, embedding_size]

        m = len(tokenizer.encode(preprompts[num_shots]))  # length of the prompt up to the token

        modified_embedding = probe_subtractor(coeff, prompt_emb_tensor[m], token.lstrip().lower()[0])

        prompt_emb_tensor[m] = modified_embedding

        # Get the logits of the next possible tokens after the prompt
        with torch.no_grad():  # It's important to use no_grad() here to prevent memory leakage
            outputs = GPTmodel(inputs_embeds=prompt_emb_tensor.unsqueeze(0))
            next_token_logits = outputs.logits[0, -1, :]  # Get the logits for the next token

        # Get the token id with the highest logit
        top_token_id = next_token_logits.argmax()

 
        # Get the highest logit value
        max_logit_value = next_token_logits.max()  # This retrieves the maximum logit value

        # Get the token id with the highest logit
        output = token_strings[top_token_id]  # You may need to adjust this if your token_strings isn't a direct id-to-string mapping

        # Generate output sequence
        generated_output = GPTmodel.generate(inputs_embeds=prompt_emb_tensor.unsqueeze(0), max_length=prompt_length + 20, temperature=temperature, pad_token_id=tokenizer.eos_token_id)  # or any appropriate max_length

        # Decode the output sequence (including the original prompt)
        full_output_text = tokenizer.decode(generated_output[0], skip_special_tokens=True)

        print(f"FULL OUTPUT: {full_output_text}\n")

        if output.lower() == token.lstrip().lower()[0]:
            pred_correct += 1
        else:
            pred_wrong +=1
            failure_tokens.append(token)
        
        single_token_results_dict = {}
        single_token_results_dict["index"] = idx
        single_token_results_dict["token"] = token
        single_token_results_dict["first letter"] = token.lstrip()[0]
        single_token_results_dict["prompt prediction"] = output
        single_token_results_dict["prediction logit"] = max_logit_value.item()

        results_dict["predictions"].append(single_token_results_dict)

        print(f"PROMPT PREDICTION: {output}")
        print(f"PROMPT CORRECT: {pred_correct}/{pred_correct + pred_wrong} ({100*pred_correct/(pred_correct + pred_wrong):.2f}%)")

        print('\n')

    if use_wandb:
        wandb.log({"results": results_dict})

    df = pd.DataFrame(results_dict["predictions"])
    df["number of shots"] = results_dict["number of shots"]
    df["prompt template"] = results_dict["prompt template"]
    df["intervention type"] = results_dict["intervention type"]
    df["intervention scale"] = results_dict["intervention scale"]

    #print(failure_tokens)

    return df, results_dict
