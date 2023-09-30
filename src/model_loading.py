import os 
import torch
from transformers import AutoTokenizer, GPTJForCausalLM

def load_or_download_model(model_name="EleutherAI/gpt-j-6B", device = "cpu"):
    '''
    # Call the function with desired model name
    tokenizer, GPTmodel, embeddings = load_or_download_model(
        model_name="gpt2", device = "cpu")
    '''
    if not os.path.exists(f'./models/{model_name}'):
        os.makedirs(f'./models/{model_name}', exist_ok=True)

    TOKENIZER_PATH = f"./models/{model_name}/tokenizer.pt"
    MODEL_PATH = f"./models/{model_name}/model.pt"
    EMBEDDINGS_PATH = f"./models/{model_name}/embeddings.pt"

    # Load or Download Tokenizer
    if os.path.exists(TOKENIZER_PATH):
        print(f'Loading {model_name} tokenizer from local storage...')
        tokenizer = torch.load(TOKENIZER_PATH)
    else:
        print(f'Downloading {model_name} tokenizer...')
        tokenizer = AutoTokenizer.from_pretrained(f"{model_name}")
        torch.save(tokenizer, TOKENIZER_PATH)

    # Load or Download Model
    if os.path.exists(MODEL_PATH):
        print(f'Loading {model_name} model from local storage...')
        GPTmodel = torch.load(MODEL_PATH).to(device)
    else:
        print(f'Downloading {model_name} model...')
        GPTmodel = GPTJForCausalLM.from_pretrained(f"{model_name}").to(device)
        torch.save(GPTmodel, MODEL_PATH)
        
    GPTmodel.eval()

    # Save or Load Embeddings
    if os.path.exists(EMBEDDINGS_PATH):
        print(f'Loading {model_name} embeddings from local storage...')
        embeddings = torch.load(EMBEDDINGS_PATH).to(device)
    else:
        embeddings = GPTmodel.transformer.wte.weight.to(device)
        torch.save(embeddings, EMBEDDINGS_PATH)
        print(f"The {model_name} 'embeddings' tensor has been saved.")

    return tokenizer, GPTmodel, embeddings

