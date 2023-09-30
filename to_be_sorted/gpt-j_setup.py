# Check if model and tokenizer exist in Google Drive
GDRIVE_PATH = os.path.join(GD_PATH, "My Drive/SpellingMiracleCollab")  # Adjust path as necessary
MODEL_PATH = os.path.join(GDRIVE_PATH, "gpt-j_model")
TOKENIZER_PATH = os.path.join(GDRIVE_PATH, "gpt-j_tokenizer")
EMBEDDINGS_PATH = os.path.join(GDRIVE_PATH, "gpt-j_embeddings")

device = 'cpu' 

# Load or Download Tokenizer
if os.path.exists(TOKENIZER_PATH):
    print('Loading GPT-J tokenizer from Google Drive...')
    tokenizer = torch.load(TOKENIZER_PATH)
else:
    print('Downloading GPT-J tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
    torch.save(tokenizer, TOKENIZER_PATH)

# Load or Download Model
if os.path.exists(MODEL_PATH):
    print('Loading GPT-J model from Google Drive...')
    GPTmodel = torch.load(MODEL_PATH).to(device)
else:
    print('Downloading GPT-J model...')
    GPTmodel = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B").to(device)
    torch.save(GPTmodel, MODEL_PATH)
    
GPTmodel.eval()

# Save or Load Embeddings
if os.path.exists(EMBEDDINGS_PATH):
    print('Loading GPT-J embeddings from Google Drive...')
    embeddings = torch.load(EMBEDDINGS_PATH).to(device)
else:
    embeddings = GPTmodel.transformer.wte.weight.to(device)
    torch.save(embeddings, EMBEDDINGS_PATH)
    print("The GPT-J 'embeddings' tensor has been saved.")