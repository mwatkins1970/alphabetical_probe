# This finds the closest top_k tokens to a given embedding vector 'emb' (this could be one row of a shape-(26,4096) probes tensor)

def closest_tokens(emb, top_k):
    
    # Compute cosine similarity and subtract from 1 to get distance
    # Higher similarity means lower distance
    distances = 1 - F.cosine_similarity(embeddings[:50257], emb.unsqueeze(0).expand_as(embeddings[:50257]))
    
    # Get the indices of the top k closest tokens
    closest_indices = torch.argsort(distances)[:top_k]

    print(closest_indices)
    
    # Return the corresponding token strings for these indices
    closest_tokens = [token_strings[i] for i in closest_indices]
    
    return closest_tokens
