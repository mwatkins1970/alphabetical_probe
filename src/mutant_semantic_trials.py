# The idea is to mutate tokens which switch from first to second letter in first letter prediction tasks,
# then pass them to GPT-J, bundled in prompts requesting definitions.

# [ This definitely works! Change GPU type if you have device-related problems running it in Colab ] 

import numpy as np
import torch 

from probe_subtractor import probe_subtractor

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

def mutant_semantic_trials_runner(switch_words, token_strings, GPTmodel, tokenizer, embeddings):

    def list_to_dict(input_list):
        output_dict = {}
        for item in input_list:
            word, number = item  # unpack the tuple
            if number > 0:
                if number in output_dict:  # number = 0 means the prompt methods fails on unmutated tokens
                    # If the number is already a key in the dictionary, append the word to the list
                    output_dict[number].append(word)
                else:
                    # If the number is not a key in the dictionary, create a new list with this word
                    output_dict[number] = [word]
        return output_dict

    switch_dict = list_to_dict(switch_words)  # builds a dictionary with switch coeff's as keys, lists of word-tokens as values

    # Print the resulting dictionary with keys in ascending order
    for key in sorted(switch_dict):
        print(f"{key}: {switch_dict[key]}")

    coeffs = list(switch_dict.keys())
    coeffs.sort() # coeffs is now the list, in ascendng order, of coeff values which are keys in the swwitch_dict

    temperature = 0.00000000001

    for coeff in coeffs:

        words = switch_dict[coeff]

        print(f"\nThese words 'lose' their first letter when the mutation coefficient has reached {coeff}:")
        print(words)

        for word in words:

            word_idx = token_strings.index(word)
            print(f"\nTOKEN: '{word}' (token index {word_idx})")

            original_embedding = embeddings[word_idx]

            prompt = f"A typical definition of the word '{word}' would be"
            print(f"PROMPT: {prompt}\n")

            # Get the original word embedding layer from the GPT model
            original_wte = GPTmodel.get_input_embeddings()

            # Generate text with original, unmutated token (control: should match output for mutation coefficient 0)
            input_prompt = f"A typical definition of '{word}' is"
            input_ids = tokenizer.encode(input_prompt, return_tensors='pt')
            output_ids = GPTmodel.generate(
            input_ids,
            attention_mask=torch.ones_like(input_ids),
            max_length=40,
            pad_token_id=tokenizer.eos_token_id  # setting pad_token_id to eos_token_id
            )
            output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

            print('PROMPTING WITH IDs, NOT EMBEDDINGS AND USING ORIGINAL, UNMUTATED TOKEN (CONTROL):')
            print(f"OUTPUT: {output_text}")






            from token_utils import closest_tokens
            closest_100 = closest_tokens(original_embedding.unsqueeze(0), 100, token_strings, embeddings, 2500)
              #filtering 2500 closest-to-centroid
            print(closest_100)






            print('\nPROMPTING WITH EMBEDDINGS AND VARIOUS MUTATIONS OF THE TOKEN:')

            for k in [0, 1, 2, 5, 10, 20, 30, 40, 60, 80, 100]:
            #for k in [0, 1, 2, 5, 8, 11, 14, 17, 20, 23, 26]:

                print(f"MUTATION COEFFICIENT: {k}")

                # mutate the token embedding by projecting along the direction of the linear probe associated with the word's first letter, scaled by coeff
                new_embedding = probe_subtractor(k, original_embedding, word.lstrip().lower()[0])


                displacement = original_embedding - new_embedding
                displacement_str = np.array2string(displacement.detach().cpu().numpy(), formatter={'float_kind':lambda x: "%.4f" % x})

                print(f"EMBEDDING DISPLACEMENT: {displacement_str}")

                # Create a custom embedding using the original embedding layer
                custom_embedding = CustomEmbedding(original_wte)

                # Set the custom embedding as the model's new embedding layer
                GPTmodel.set_input_embeddings(custom_embedding)

                # Add the embedding modification
                custom_embedding.add_modification(word_idx, new_embedding)

                # Generate text with altered token
                input_prompt = f"A typical definition of '{word}' is"
                input_ids = tokenizer.encode(input_prompt, return_tensors='pt')
                altered_output_ids = GPTmodel.generate(
                input_ids,
                attention_mask=torch.ones_like(input_ids),
                max_length=25,
                pad_token_id=tokenizer.eos_token_id  # setting pad_token_id to eos_token_id
                )

                altered_output_text = tokenizer.decode(altered_output_ids[0], skip_special_tokens=True).split('\n')[0]

                print(f"OUTPUT: {altered_output_text}\n")
                # Remove the embedding modification to revert to the original token
                custom_embedding.remove_modifications()

                # Reset the original embeddings on the model after removing modifications
                GPTmodel.set_input_embeddings(original_wte)


                closest_100 = closest_tokens(new_embedding.unsqueeze(0), 100, token_strings, embeddings, 2500)
                print(f"CLOSEST 100 TOKENS (CTC FILTERED): {closest_100}")
                print('\n')

