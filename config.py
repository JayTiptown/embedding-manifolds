import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    d_model = 256
    n_layers = 4
    n_heads = 4
    d_ff = 1024
    vocab_size = 256
    max_seq_len = 256
    dropout = 0.1
    
    batch_size = 32
    num_epochs = 10
    lr = 3e-4
    muon_lr = 0.02
    
    eval_interval = 100
    eval_iters = 50
    
    torch = __import__('torch')
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    
    print(f"Config initialized - Using device: {device}")
    
    wandb_api_key = os.getenv('WANDB_API_KEY')
    wandb_entity = os.getenv('WANDB_ENTITY')
    wandb_project = os.getenv('WANDB_PROJECT')
    
    seed = 42

