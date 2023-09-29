# This finds the closest k tokens to the vector you end up with when you add together the probe vectors for some specified list of letters 'letter_list'
# coeff is currently set to 0, but if you increase it, the probe-vector-sum is further adjusted: the probe vectors for all letters NOT in the list are subtracted from it,
# scaled by coeff. Experiments show that as coeff approaches 1, tokens get shorter, there are less extraneous letters. You can study how coeff affects the
# effectiveness of this "alphabetic vector arithmetic" in the next cell, using a metric I introduce for how well tokens match the requested letter components.

import torch.nn.functional as F

def proportion_of_letters(token, letter_list):
    unique_letters = set(letter_list)
    count = sum([1 for letter in unique_letters if letter in token.lower()])
    return f"{count}/{len(unique_letters)}"

def net_proportion_of_letters(token, letter_list):
    count = sum([1 for char in token.lstrip().lower() if char in letter_list]) + sum([-1 for char in token.lstrip().lower() if char not in letter_list])
    return f"{count}/{len(token.lstrip())}"

def closest_tokens(letter_list, coeff, distance_type="cosine"):

    probe_sum = torch.zeros(4096)
    for letter in letter_list:
        probe_sum += probe_weights_tensor[ord(letter.lower()) - 97]

    # This bit subtracts probes for any letters NOT in the list. It results in much shorter tokens for the most part.
    for char in "abcdefghijklmnopqrstuvwxyz":
        if char not in letter_list:
          probe_sum = probe_sum - coeff * probe_weights_tensor[ord(char) - 97]

    # Compute distances
    if distance_type == "cosine":
        distances = 1 - F.cosine_similarity(embeddings[:50257], probe_sum.unsqueeze(0).expand_as(embeddings[:50257]))
    elif distance_type == "euclidean":
        distances = torch.norm(embeddings[:50257] - probe_sum, dim=1)
    else:
        raise ValueError("Invalid distance type. Choose 'cosine' or 'euclidean'")

    # Get the indices of the top k closest tokens
    closest_indices = torch.argsort(distances)[:top_k]

    # Return the corresponding token indices, strings, and their distances for these indices
    closest_token_indices = [i.item() for i in closest_indices]
    closest_token_distances = [distances[i].item() for i in closest_indices]
    closest_token_strings = [token_strings[i] for i in closest_indices]

    #just testing with random all-rom tokens... ERASE THIS IMMEDIATELY
    #closest_token_strings = [token_strings[i] for i in random.choices(all_rom_token_indices, k = top_k)]

    return list(zip(closest_token_indices, closest_token_strings, closest_token_distances))

top_k = 100
strg = "crayon"
letter_list = [char for char in strg]
coeff = 0.2

closest_results = closest_tokens(letter_list, coeff, distance_type="cosine")

# Initialize a variable to store the sum of numerators from 'prop' values
total_prop_numerator = 0
total_net_prop_numerator = 0

total_letters = 0

for (index, token, distance) in closest_results:
    prop = proportion_of_letters(token, letter_list)
    net_prop = net_proportion_of_letters(token, letter_list)
    # Extract the numerator from the 'prop' string and add it to the total
    numerator = int(prop.split('/')[0])
    net_numerator = int(net_prop.split('/')[0])
    total_prop_numerator += numerator
    total_net_prop_numerator += net_numerator
    total_letters += len(token.lstrip())

# Compute the average of the 'prop' values
average_prop = total_prop_numerator / total_letters
average_net_prop = total_net_prop_numerator / total_letters

#print(f"Subtractive coefficient: {coeff}; Average proportion of letters present: {average_prop/len(set(letter_list)):.4f}")
print(f"Subtractive coefficient: {coeff}; Average net proportion of letters present: {average_net_prop/len(set(letter_list)):.4f}")

#Testing the function
print('-'*68)  # Print a separator line
#print(f"{'Index':<10} {'Token':<20} {'Distance':<15} {'Proportion of letters'}")
print(f"{'Index':<10} {'Token':<20} {'Distance':<15} {'Net proportion of letters'}")
print('-'*68)  # Print a separator line

for (index, token, distance) in closest_results:
    prop = proportion_of_letters(token, letter_list)
    net_prop = net_proportion_of_letters(token, letter_list)

    #print(f"{index:<10} {token:<20} {distance:.4f} {prop:^20}")
    print(f"{index:<10} {token:<20} {distance:.4f} {net_prop:^20}")