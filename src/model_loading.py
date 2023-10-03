import os
import torch
from transformers import AutoTokenizer, GPTJForCausalLM

def load_or_download_tokenizer(model_name):
    TOKENIZER_PATH = f"./models/{model_name}/tokenizer.pt"
    if os.path.exists(TOKENIZER_PATH):
        print(f'Loading {model_name} tokenizer from local storage...')
        return torch.load(TOKENIZER_PATH)
    else:
        print(f'Downloading {model_name} tokenizer...')
        tokenizer = AutoTokenizer.from_pretrained(f"{model_name}")
        torch.save(tokenizer, TOKENIZER_PATH)
        return tokenizer

def load_or_download_model(model_name, device):
    MODEL_PATH = f"./models/{model_name}/model.pt"
    if os.path.exists(MODEL_PATH):
        print(f'Loading {model_name} model from local storage...')
        return torch.load(MODEL_PATH).to(device)
    else:
        print(f'Downloading {model_name} model...')
        model = GPTJForCausalLM.from_pretrained(f"{model_name}").to(device)
        torch.save(model, MODEL_PATH)
        return model

def load_or_save_embeddings(model_name, device):
    EMBEDDINGS_PATH = f"./models/{model_name}/embeddings.pt"
    if os.path.exists(EMBEDDINGS_PATH):
        print(f'Loading {model_name} embeddings from local storage...')
        return torch.load(EMBEDDINGS_PATH).to(device)
    else:
        embeddings = model.transformer.wte.weight.to(device)
        torch.save(embeddings, EMBEDDINGS_PATH)
        print(f"The {model_name} 'embeddings' tensor has been saved.")
        return embeddings

def load_or_download_model_tok_emb(model_name="EleutherAI/gpt-j-6B", device="cpu"):
    if not os.path.exists(f'./models/{model_name}'):
        os.makedirs(f'./models/{model_name}', exist_ok=True)

    tokenizer = load_or_download_tokenizer(model_name)
    embeddings = load_or_save_embeddings(GPTmodel, model_name, device)
    GPTmodel = load_or_download_model(model_name, device)
    GPTmodel.eval()
    
    return tokenizer, GPTmodel, embeddings

