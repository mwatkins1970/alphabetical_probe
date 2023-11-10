# THIS SUBTRACTS OUT THE PART OF AN EMBEDDING VECTOR PARALLEL TO A PROBE, PRODUCING SOMETHING WHICH IS 
# THEREFORE ORTHOGONAL (0 cosine similarity) TO IT

import torch
import os
import requests

def load_probe_weights_tensor(filename='pos1_probe_weights_tensor.pt'):
    # Check if the file exists locally; if not, download it
    if not os.path.isfile(filename):
        url = 'https://github.com/mwatkins1970/alphabetical_probe/raw/main/pos1_probe_weights_tensor.pt'
        r = requests.get(url)
        with open(filename, 'wb') as f:
            f.write(r.content)

    # Load the tensor using PyTorch
    probe_weights_tensor = torch.load(filename)
    return probe_weights_tensor


def probe_subtractor(coeff, emb, letter):
    if len(letter) == 1 and letter.lower() in "abcdefghijklmnopqrstuvwxyz":
        probe_weights_tensor = load_probe_weights_tensor()
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
