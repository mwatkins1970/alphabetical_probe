import torch
import wandb
import pandas as pd
from find_closest_probe import find_closest_probe
from probe_subtractor import probe_subtractor
from probe_subtractor import probe_subtractor_alt

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

        ix = tokenizer.encode(prompt)

        #print(f"\nTOKEN INDEX LIST: {ix}")

        ix_tensor = torch.tensor(ix).unsqueeze(0)
        prompt_emb_tensor = embeddings[ix_tensor]

        # Calculate the index k
        k = len(tokenizer.encode(preprompts[num_shots]))  # length of the prompt up to the token
        #print(f"Modifying embedding at index: {k}")

        # Modify the embedding at index k using probe_subtractor
        # the first one projects onto the probe's orthogonal complement
        # the second one projects ACROSS it
        #modified_embedding = probe_subtractor(prompt_emb_tensor[0, k], token.lstrip().lower()[0])
        modified_embedding = probe_subtractor_alt(coeff, prompt_emb_tensor[0, k], token.lstrip().lower()[0])

        #print('\n')
        #print(f"ORIGINAL EMBEDDING FOR TOKEN: {prompt_emb_tensor[0, k]}")
        #print(f"DIFFERENCE BETWEEN THESE: {prompt_emb_tensor[0, k] - modified_embedding}")
        #print(f"MODIFIED EMBEDDING FOR TOKEN: {modified_embedding}")

        # Replace the embedding at index k in prompt_emb_tensor with the modified embedding
        prompt_emb_tensor[0, k] = modified_embedding

        #print(f"\nNEW PROMPT EMBEDDINGS TENSOR: {prompt_emb_tensor}")

        model_out = GPTmodel(inputs_embeds=prompt_emb_tensor, return_dict=True)

        logits = model_out.logits
        scaled_logits = logits / temperature
        probabilities = torch.nn.functional.softmax(scaled_logits, dim=-1)

        # Sample a token from the adjusted distribution
        probabilities_last_token = probabilities[0, -1, :]
        sampled_token_id = torch.multinomial(probabilities_last_token, 1)

        sampled_token_id = sampled_token_id.item()

        output = token_strings[sampled_token_id]

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
