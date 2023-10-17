import torch
import wandb
import pandas as pd
from find_closest_probe import find_closest_probe
from probe_subtractor import probe_subtractor
from transformers import GPTJForCausalLM

device = "cpu"

temperature = 0.00000000001

class CustomEmbedding(torch.nn.Module):
    def __init__(self, original_embedding):
        super().__init__()
        self.original_embedding = original_embedding  # changed from self.embedding to self.original_embedding
        self.modifications = {}

    def add_modification(self, token_id, new_embedding):
        self.modifications[token_id] = new_embedding

    def remove_modifications(self):
        self.modifications.clear()  # This empties the dictionary of modifications        

    def forward(self, input_ids=None):
        # Get the original embeddings
        original_embeddings = self.original_embedding(input_ids)
        
        # Apply any modifications
        for token_id, new_embedding in self.modifications.items():
            mask = (input_ids == token_id)
            original_embeddings[mask] = new_embedding
        
        return original_embeddings


def nu_comparative_first_letter_evals_runner(GPTmodel, tokenizer, embeddings, token_strings, all_rom_token_gt2_indices, token_index, num_shots, coeff):

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
    results_dict["intervention type"] = 'orthogonal projection'
    results_dict["intervention scale"] = coeff
    results_dict["predictions"] = []

    # iterate through our list of token indices (each token will have its embedding mutated before being inserted into the prompt template and forward-passed)

    token = token_strings[token_index]

    input_prompt = preprompts[num_shots] + token + '''" begins with the letter "'''
    print(f"PROMPT:\n{input_prompt}")

    # Get the original word embedding layer from the GPT model
    original_wte = GPTmodel.get_input_embeddings()

    original_embedding = embeddings[token_index]

    new_embedding = probe_subtractor(coeff, original_embedding, token.lstrip().lower()[0])

    # Create a custom embedding using the original embedding layer
    custom_embedding = CustomEmbedding(original_wte)

    # Set the custom embedding as the model's new embedding layer
    GPTmodel.set_input_embeddings(custom_embedding)

    # Add the embedding modification
    custom_embedding.add_modification(token_index, new_embedding)

    # Generate logits with the altered token embedding
    input_ids = tokenizer.encode(input_prompt, return_tensors='pt')
    with torch.no_grad():  # No need to track gradients when doing inference
        outputs = GPTmodel(input_ids=input_ids, attention_mask=torch.ones_like(input_ids))
        logits = outputs.logits  # shape is batch_size x sequence_length x vocab_size

    # We only need the logits for the last token in the sequence
    last_token_logits = logits[0, -1, :]

    # Find the token ID with the highest logit (greedy selection, equivalent to temperature 0)
    predicted_token_id = torch.argmax(last_token_logits).item()
    highest_logit_value = last_token_logits[predicted_token_id].item()

    # Convert the token ID back to a string token
    predicted_token = tokenizer.decode([predicted_token_id])

    # Remove the embedding modification to revert to the original token
    custom_embedding.remove_modifications()

    # Reset the original embeddings on the model after removing modifications
    GPTmodel.set_input_embeddings(original_wte)


    print(f"OUTPUT: {predicted_token}\n")
    
    single_token_results_dict = {}
    single_token_results_dict["index"] = token_index
    single_token_results_dict["token"] = token
    single_token_results_dict["first letter"] = token.lstrip()[0]
    single_token_results_dict["prompt prediction"] = predicted_token
    single_token_results_dict["prediction logit"] = highest_logit_value  
    
    results_dict["predictions"].append(single_token_results_dict)

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
