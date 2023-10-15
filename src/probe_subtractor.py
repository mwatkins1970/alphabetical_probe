# THIS SUBTRACTS OUT THE PART OF AN EMBEDDING VECTOR PARALLEL TO A PROBE, PRODUCING SOMETHING WHICH IS 
# THEREFORE ORTHOGONAL (0 cosine similarity) TO IT

import torch

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

        emb_parallel = (torch.dot(emb, letter_probe) / torch.dot(letter_probe, letter_probe)) * letter_probe

        return emb - emb_parallel # This is the projection INTO the orthogonal complement of the probe.


# Rather than projecting onto the orthogonal complement of the probe vector, this projects ACROSS it.

def probe_subtractor_alt(coeff, emb, letter):
    if len(letter) == 1 and letter.lower() in "abcdefghijklmnopqrstuvwxyz":
        probe_weights_tensor = torch.load('/content/Drive/My Drive/SpellingMiracleCollab/pos1_probe_weights_tensor.pt')
        probe_idx = ord(letter) - 97
        device = emb.device
        probe_weights_tensor = probe_weights_tensor.to(device)
        letter_probe = probe_weights_tensor[probe_idx]

        emb_parallel = (torch.dot(emb, letter_probe) / torch.dot(letter_probe, letter_probe)) * letter_probe
        emb_perp = emb - emb_parallel

        emb_refl = emb - coeff * emb_parallel

        return emb_refl # This is the original embedding for coeff = 0;
                        # the projection into the orthogonal complement for coeff = 1
                        # the reflection across the orthogonal complement for coeff = 2
                        # "reflection even further back" for coeff > 2
