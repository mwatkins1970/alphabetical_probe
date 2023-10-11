# THIS SUBTRACTS OUT THE PART OF AN EMBEDDING VECTOR PARALLEL TO A PROBE, PRODUCING SOMETHING WHICH IS 
# THEREFORE ORTHOGONAL (0 cosine similarity) TO IT

def probe_subtractor(emb, letter):
    if len(letter) == 1 and letter.lower() in "abcdefghijklmnopqrstuvwxyz":
        # probe_weights_tensor = all_probe_training_runner(embeddings, all_rom_token_indices, token_strings, probe_type = 'linear', use_wandb = True, criteria_mode = "pos1")
        probe_weights_tensor = torch.load('/content/Drive/My Drive/SpellingMiracleCollab/pos1_probe_weights_tensor.pt')

        probe_idx = ord(letter) - 97

        # Get the device of the emb tensor
        device = emb.device
        
        # Ensure the probe_weights_tensor is on the same device as emb
        probe_weights_tensor = probe_weights_tensor.to(device)

        # Ensure letter_probe is derived from the device-matched tensor
        letter_probe = probe_weights_tensor[probe_idx]

        return emb - (torch.dot(emb, letter_probe) / torch.dot(letter_probe, letter_probe)) * letter_probe
