import re
import random
import torch

def get_training_data(
    num_samples, 
    embeddings, 
    all_rom_token_gt2_indices,
    token_strings):

    # Fetch index pairs for tokens that match the required pattern
    positive_indices = [(idx1, idx2) for idx1 in all_rom_token_gt2_indices for idx2 in all_rom_token_gt2_indices if idx1 != idx2 and token_strings[idx1] == token_strings[idx2][:len(token_strings[idx1])]]

    # Fetch indices for tokens that do not begin this way
    # (by taking a set difference and then converting back to a list)
    all_pairs = [(idx1, idx2) for idx1 in all_rom_token_gt2_indices for idx2 in all_rom_token_gt2_indices if idx1 != idx2]
    negative_indices = [pair for pair in all_pairs if pair not in positive_indices]

    # Randomly sample from positive and negative indices to balance the dataset
    num_positive = min(num_samples // 2, len(positive_indices))
    num_negative = num_samples - num_positive

    sampled_positive_indices = random.sample(positive_indices, num_positive)
    sampled_negative_indices = random.sample(negative_indices, num_negative)

    # Combine sampled indices
    sampled_indices = sampled_positive_indices + sampled_negative_indices
    random.shuffle(sampled_indices)  # Shuffle combined indices for randomness in training
 
    # Create concatenated embeddings for each pair of indices
    all_embeddings = torch.stack([torch.cat((embeddings[idx1], embeddings[idx2]), dim=0) for (idx1, idx2) in sampled_indices])

    # Create labels: 1 if the pair is in positive_indices, 0 otherwise
    all_labels = [1 if (idx1, idx2) in positive_indices else 0 for (idx1, idx2) in sampled_indices]

    return all_embeddings.clone().detach(), torch.tensor(all_labels).clone().detach()