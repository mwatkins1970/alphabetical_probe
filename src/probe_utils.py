# This separates all-Roman tokens into those whose first letter can/not be predicted successfully
# by looking at closest "starting letter" linear probe 

import torch 

def find_closest_probe(embedding_vector, probes_tensor):    
    # probes_tensor should be shape-(26,4096), with rows corresponding to alphabet

    # Check if all rows are zeros
    if torch.all(probes_tensor == 0):
        return '_'

    # Create a mask to identify non-zero rows in probes_tensor
    non_zero_mask = torch.any(probes_tensor != 0, dim=1)
    
    # Filter out zero rows from probes_tensor
    filtered_probes_tensor = probes_tensor[non_zero_mask]
    
    # Compute cosine similarity
    cosine_sim = F.cosine_similarity(embedding_vector.unsqueeze(0), filtered_probes_tensor)
    
    # Get the closest value and index
    top_value, top_index = torch.topk(cosine_sim, 1)
    
    # Convert the filtered tensor index to the original tensor index
    original_index = torch.arange(probes_tensor.size(0))[non_zero_mask][top_index]
    
    # Convert index to letter and build result list
    closest_probe = chr(original_index.item() + ord('a'))
    
    return closest_probe

def predict_first_letter_with_probes(
        embeddings, starting_letter_probe_weights_tensor,
        all_rom_token_indices, token_strings):

    successful_list = []
    unsuccessful_list = []

    for j in all_rom_token_indices:
        nearest_starting_letter_probe = find_closest_probe(embeddings[j], starting_letter_probe_weights_tensor)
        if nearest_starting_letter_probe == token_strings[j].lstrip().lower[0]:
            successful_list.append[j]
        else:
            unsuccessful_list.append[j]

    print(f"Number of successful predictions: {len(successful_list)}/{len(all_rom_token_indices)}")
    print(f"Number of unsuccessful predictions: {len(unsuccessful_list)}/{len(all_rom_token_indices)}")