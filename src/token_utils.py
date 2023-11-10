# This finds the closest top_k tokens to a given embedding vector 'emb' (this could be one row of a shape-(26,4096) probes tensor)

import os
import random
import json
import pickle
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def token_setup(tokenizer):
    token_strings = [tokenizer.decode([i]) for i in range(50257)]
    num_tokens = len(token_strings)
    print(f"There are {num_tokens} tokens.")

    all_rom_tokens = []			#initialise list of all-roman tokens
    all_rom_token_indices = []

    for i in range(num_tokens):
      all_rom = True                       # Is token_string[i] all roman characters? Assume to begin that it is.
      for ch in range(len(token_strings[i])):
        if token_strings[i][ch] not in ' abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ':
          all_rom = False
      if all_rom == True and token_strings[i] not in [' ', '  ', '   ', '    ', '     ', '      ', '       ', '        ']:  # eliminate all the all-space tokens
        all_rom_tokens.append(token_strings[i])
        all_rom_token_indices.append(i)

    print(f"There are {len(all_rom_tokens)} all-Roman tokens.")

    all_rom_token_gt2_indices = [idx for idx in all_rom_token_indices if len(token_strings[idx].strip()) > 2]
    print(f"There are {len(all_rom_token_gt2_indices)} all-Roman tokens with more than two letters.")

    return token_strings, all_rom_token_indices, all_rom_token_gt2_indices


# THIS MAKES A LIST OF THE 50257 SIGNIFICANT TOKENS, AS WELL AS  SUB-LISTS OF ALL-ROMAN TOKENS
def load_token_strings_etc(tokenizer):

		num_tokens = len(token_strings)

		# The leading spaces in these token strings are weird, they somehow delete themselves and the character before whatever they get appended to
		#...so you got stuff like "The string'petertodd' spelled in all capital letters..."
		#...rather than "The string ' petertodd' spelled is..." as it should be
		# This seems to be an easy fix. Just go through the lists and replace the first character with an actual space!
		for token in token_strings:
			token = " " + token[1:]

		print(f"There are {num_tokens} tokens.")

		all_rom_tokens = []			#initialise list of all-roman tokens
		all_rom_token_indices = []

		for i in range(num_tokens):
			all_rom = True                       # Is token_string[i] all roman characters? Assume to begin that it is.
			for ch in range(len(token_strings[i])):
				if token_strings[i][ch] not in ' abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ':
					all_rom = False
			if all_rom == True and token_strings[i] not in [' ', '  ', '   ', '    ', '     ', '      ', '       ', '        ']:
				all_rom_tokens.append(token_strings[i])
				all_rom_token_indices.append(i)

		all_rom_token_gt2_indices = [idx for idx in all_rom_token_indices if len(token_strings[idx].lstrip()) > 2]


		print(f"There are {len(all_rom_tokens)} all-Roman tokens.")
		print(f"There are {len(all_rom_token_gt2_indices)} all-Roman tokens with more than two letters.")
	
		return token_strings, all_rom_tokens, all_rom_token_indices, all_rom_token_gt2_indices


def closest_tokens(emb, top_k, token_strings, embeddings, filter_CTC_num):
    # 'filter_CTE_num is the number of "closest to centroid" tokens (which are also closest to everything else) you want to filter. 1000 works well.

    if filter_CTC_num > 0:

        #print(f"FILTERING {filter_CTC_num} CTC TOKENS")

        subset_embeddings = embeddings[:50257]
        centroid = torch.mean(subset_embeddings, dim=0)
        # Compute L2 distance from each vector to the centroid
        # Euclidean distance can be calculated by taking the norm of the difference
        distances = torch.norm(subset_embeddings - centroid, dim=1)
        #Rank by distance (get the sorted indices)
        sorted_indices = torch.argsort(distances)
        # Convert the tensor to a Python list
        sorted_indices_list = sorted_indices.tolist()[:filter_CTC_num]
        set_of_CTC_token_indices = set(sorted_indices_list)
    else:
        set_of_CTC_token_indices = set()



    # Make sure all tensors are on the same device as `emb`
    embeddings = embeddings.to(emb.device)
    
    # Compute cosine similarity and subtract from 1 to get distance
    # Higher similarity means lower distance
    distances = 1 - F.cosine_similarity(embeddings[:50257], emb)


    # Get the indices of the top k closest tokens
    closest_indices = torch.argsort(distances)
    filtered_closest_indices = [idx.item() for idx in closest_indices if idx.item() not in set_of_CTC_token_indices]
    closest_indices_list = filtered_closest_indices[:top_k]
    
    # Return the corresponding token strings for these indices
    closest_tokens = [token_strings[i] for i in closest_indices_list]
    
    return closest_tokens


def closest_tokens_recentred(emb, top_k, token_strings, embeddings):

    # Make sure all tensors are on the same device as `emb`
    embeddings = embeddings.to(emb.device)

    mean_emb = torch.mean(embeddings[:50257], dim=0)

    adjusted_emb = emb - mean_emb 
    adjusted_embeddings = embeddings[:50257] - mean_emb.unsqueeze(0)


    # Compute cosine similarity and subtract from 1 to get distance

    distances = 1 - F.cosine_similarity(adjusted_embeddings, adjusted_emb)


    # Get the indices of the top k closest tokens
    closest_indices = torch.argsort(distances)[:top_k]

    #print(closest_indices)
    
    # Return the corresponding token strings for these indices
    closest_tokens = [token_strings[i] for i in closest_indices]
    
    return closest_tokens


# Given a shape (26,4096) tensor (typically staring-letter probes), this
# finds the closest k probes to the embedding vector and returns them 
# as a zipped list, giving corresponding letters and cosine distances


def euc_distance_from_centroid(emb, embeddings):
    mean_emb = torch.mean(embeddings[:50257], dim=0)
    euc_distance = torch.norm(emb - mean_emb, p=2)
    return euc_distance


def find_top_k_letters(embedding_vector, letter_tensor, k):
    # Compute cosine similarity
    cosine_sim = F.cosine_similarity(embedding_vector.unsqueeze(0), letter_tensor)
    # Get the top k indices and values
    top_k_values, top_k_indices = torch.topk(cosine_sim, k)
    # Convert indices to letters and build result list
    top_k_letters = [(chr(idx.item() + ord('a')), 1 - value.item()) for idx, value in zip(top_k_indices, top_k_values)]
    return top_k_letters
