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