import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    d_model = 256
    n_layers = 4
    n_heads = 4
    d_ff = 1024
    vocab_size = 10000
    max_seq_len = 256
    dropout = 0.1
    
    batch_size = 32
    num_epochs = 10
    lr = 3e-4
    muon_lr = 0.02
    
    eval_interval = 100
    eval_iters = 50
    
    device = 'cuda' if __import__('torch').cuda.is_available() else 'cpu'
    
    wandb_api_key = os.getenv('WANDB_API_KEY')
    wandb_entity = os.getenv('WANDB_ENTITY')
    wandb_project = os.getenv('WANDB_PROJECT')
    
    seed = 42

