import pytest 

from src.probe_training import probe_training_runner
from src.model_loading import load_or_download_tokenizer, load_or_save_embeddings, load_or_download_model
from src.letter_token_utils import (
    get_token_strings,
    get_all_rom_tokens,
)

@pytest.fixture()
def model():
    model = load_or_download_model(model_name="gpt2", device="cpu")
    return model

@pytest.fixture()
def tokenizer():
    tokenizer = load_or_download_tokenizer(model_name="gpt2")
    return tokenizer

@pytest.fixture()
def embeddings(model):
    embeddings = load_or_save_embeddings(model, model_name="gpt2", device="cpu")
    return embeddings

def test_probe_training_runner(embeddings, tokenizer):

    token_strings = get_token_strings(tokenizer)
    _, all_rom_token_indices = get_all_rom_tokens(token_strings)

    all_rom_token_gt2_indices = [idx for idx in all_rom_token_indices if len(token_strings[idx].lstrip()) > 2]


    probe_weights_tensor = probe_training_runner(
        embeddings, 
        all_rom_token_gt2_indices,
        token_strings,
        alphabet="ABC",
        )
    

    assert probe_weights_tensor is not None