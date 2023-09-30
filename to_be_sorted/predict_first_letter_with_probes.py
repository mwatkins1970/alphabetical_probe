# This separates all-Roman tokens into those whose first letter can/not be predicted successfully
# by looking at closest "starting letter" linear probe 

def predict_first_letter_with_probes(embeddings, starting_letter_probe_weights_tensor)

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