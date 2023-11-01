import torch
import torch.nn.functional as F
from collections import OrderedDict

def find_closest_probe(embedding_vector, probes_tensor):
    # probes_tensor should be shape-(26, 4096), with rows corresponding to the alphabet

    # Check if all rows are zeros
    if torch.all(probes_tensor == 0):
        return '_'

    # Create a mask to identify non-zero rows in probes_tensor
    non_zero_mask = torch.any(probes_tensor != 0, dim=1)
    
    # Filter out zero rows from probes_tensor
    filtered_probes_tensor = probes_tensor[non_zero_mask]
    
    # Ensure both tensors are on the same device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    embedding_vector = embedding_vector.to(device)
    filtered_probes_tensor = filtered_probes_tensor.to(device)
    
    # Compute cosine similarity
    cosine_sim = F.cosine_similarity(embedding_vector.unsqueeze(0), filtered_probes_tensor)
    
    # Get the closest value and index
    top_value, top_index = torch.topk(cosine_sim, 1)
    
    # Create a range tensor on the same device
    range_tensor = torch.arange(probes_tensor.size(0), device=device)

    # Convert the filtered tensor index to the original tensor index
    original_index = range_tensor[non_zero_mask][top_index]
    
    # Convert index to letter and build result list
    closest_probe = chr(original_index.item() + ord('a'))
    
    return closest_probe

# This finds the farthest probe, or equivalently, the closest 'anti-probe'
def find_closest_antiprobe(embedding_vector, probes_tensor):    
    # probes_tensor should be shape-(26,4096), with rows corresponding to alphabet

    # Check if all rows are zeros
    if torch.all(probes_tensor == 0):
        return '_'

    # Create a mask to identify non-zero rows in probes_tensor
    non_zero_mask = torch.any(probes_tensor != 0, dim=1)
    
    # Filter out zero rows from probes_tensor
    filtered_probes_tensor = probes_tensor[non_zero_mask]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    embedding_vector = embedding_vector.to(device)
    filtered_probes_tensor = filtered_probes_tensor.to(device)  # Corrected line

    # Compute cosine similarity with the negative of the embedding vector
    cosine_sim = F.cosine_similarity((-1) * embedding_vector.unsqueeze(0), filtered_probes_tensor)
    
    # Get the closest value and index
    top_value, top_index = torch.topk(cosine_sim, 1)

    # Create a range tensor on the same device
    range_tensor = torch.arange(probes_tensor.size(0), device=device)
    
    # Convert the filtered tensor index to the original tensor index
    original_index = range_tensor[non_zero_mask][top_index]
    
    # Convert index to letter and build result list
    closest_antiprobe = chr(original_index.item() + ord('a'))
    
    return closest_antiprobe


# Returns a sorted dictionary of distances of the 26 probes encoded in weights_tensor (shape [26,4096]) to the [4096] tensor (embedding) emb
def probe_distances(emb, weights_tensor):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Adjusting the shape of 'emb'
    emb = emb.squeeze().to(device)

    distance_dict = {}
    for i, probe in enumerate(weights_tensor):
        if probe.nelement() == 0:
            continue

        probe = probe.to(device)

        # Computing the similarity
        similarity = F.cosine_similarity(emb.unsqueeze(0), probe.unsqueeze(0)).squeeze()

        if similarity.nelement() != 1:
            raise ValueError("Expected a single element tensor for similarity.")

        distance = 1 - similarity.item()
        distance_dict[chr(i + ord('A'))] = distance

    sorted_distance_dict = {k: v for k, v in sorted(distance_dict.items(), key=lambda item: item[1])}
    return sorted_distance_dict



# prints dictionary of distances from embedding to all 26 probes, and plots a bar graph
def display_closest_probes(emb, weights_tensor):

    sorted_distance_dict = probe_distances(emb, weights_tensor)

    print(sorted_distance_dict)

    # Create a bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(sorted_distance_dict.keys(), sorted_distance_dict.values())
    plt.xlabel('FL probes')
    plt.ylabel('cosine distance')
    plt.title(f"Normalized Cosine Distances from embedding to all FL Probes")
    plt.show()



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
