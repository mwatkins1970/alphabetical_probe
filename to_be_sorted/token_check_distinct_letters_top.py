# This checks tokens to see if their set of n distinct letters matches the set of n closest letter-probe directions.)

count = 0
success_list = []

# Iterate through all_rom_token_indices

for i in all_rom_token_indices:

    token = token_strings[i]
    k = len(set(token.lower())) - (1 if ' ' in token else 0)  # Adjust for spaces if present

    results = find_top_k_letters(embeddings[i], letter_tensor, k)

    # Extract the letters from the results
    top_k_letters = [item[0] for item in results]
    
    # Check if all token letters are in the top_k list
    if all(letter in top_k_letters for letter in token.lstrip().lower()):
        count +=1
        success_list.append(token)
        print(f"token index: {i}; token string: {token}; token length: {len(token.lstrip()) ")
        print(f"top k letters: {top_k_letters}")

        print('-' * 50)  # Print a separator line

print(f"A total of {count}/{len(all_rom_token_indices)} all-Roman tokens are such that their set of n distinct letters corresponds to the set of n closest letter-probe directions.")
