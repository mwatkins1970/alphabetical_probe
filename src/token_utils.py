# This finds the closest top_k tokens to a given embedding vector 'emb' (this could be one row of a shape-(26,4096) probes tensor)
import torch
import torch.functional as F

def closest_tokens(emb, top_k, token_strings, embeddings):
    
    # Compute cosine similarity and subtract from 1 to get distance
    # Higher similarity means lower distance
    distances = 1 - F.cosine_similarity(embeddings[:50257], emb.unsqueeze(0).expand_as(embeddings[:50257]))
    
    # Get the indices of the top k closest tokens
    closest_indices = torch.argsort(distances)[:top_k]

    print(closest_indices)
    
    # Return the corresponding token strings for these indices
    closest_tokens = [token_strings[i] for i in closest_indices]
    
    return closest_tokens




# Given a shape (26,4096) tensor (typically staring-letter probes), this
# finds the closest k probes to the embedding vector and returns them 
# as a zipped list, giving correspoding letters and cosine distances

def find_top_k_letters(embedding_vector, letter_tensor, k):
    # Compute cosine similarity
    cosine_sim = F.cosine_similarity(embedding_vector.unsqueeze(0), letter_tensor)
    # Get the top k indices and values
    top_k_values, top_k_indices = torch.topk(cosine_sim, k)
    # Convert indices to letters and build result list
    top_k_letters = [(chr(idx.item() + ord('a')), 1 - value.item()) for idx, value in zip(top_k_indices, top_k_values)]
    return top_k_letters