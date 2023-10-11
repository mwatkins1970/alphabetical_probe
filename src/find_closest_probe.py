# probes_tensor should be shape-(26,4096), with rows corresponding to the alphabet

def find_closest_probe(embedding_vector, probes_tensor):    
  
    import torch
    import torch.nn.functional as F

    # Check if all rows are zeros
    if torch.all(probes_tensor == 0):
        return '_', [-float('inf')] * 26

    # Create a mask to identify non-zero rows in probes_tensor
    non_zero_mask = torch.any(probes_tensor != 0, dim=1)

    # Filter out zero rows from probes_tensor
    filtered_probes_tensor = probes_tensor[non_zero_mask]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    embedding_vector = embedding_vector.to(device)
    filtered_probes_tensor = filtered_probes_tensor.to(device)

    # Compute cosine similarity
    cosine_sim = F.cosine_similarity(embedding_vector.unsqueeze(0), filtered_probes_tensor)

    # Create a tensor of size 26 filled with -infinity
    all_cosine_sim = torch.full((26,), -float('inf'), device=device)

    # Fill the computed cosine similarities in the appropriate positions of the all_cosine_sim tensor
    all_cosine_sim[non_zero_mask] = cosine_sim

    # Get the closest value and index
    top_value, top_index = torch.topk(cosine_sim, 1)

    # Convert the filtered tensor index to the original tensor index
    original_index = torch.arange(probes_tensor.size(0), device=device)[non_zero_mask][top_index]

    # Convert index to letter and build result list
    closest_probe = chr(original_index.item() + ord('a'))

    return closest_probe, all_cosine_sim.cpu().tolist()
