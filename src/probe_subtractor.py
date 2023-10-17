# THIS SUBTRACTS OUT THE PART OF AN EMBEDDING VECTOR PARALLEL TO A PROBE, PRODUCING SOMETHING WHICH IS 
# THEREFORE ORTHOGONAL (0 cosine similarity) TO IT

import torch

def probe_subtractor(coeff, emb, letter):
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
