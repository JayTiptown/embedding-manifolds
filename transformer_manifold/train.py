"""
Training loop with condition number tracking.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from tqdm import tqdm
import math
import wandb

from .models import ManifoldTransformer
from optimizers import ManifoldMuon, SphereOptimizer


class TextDataset(Dataset):
    """Simple character-level dataset."""
    def __init__(self, data, seq_len, vocab_size):
        self.data = data
        self.seq_len = seq_len
        self.vocab_size = vocab_size
    
    def __len__(self):
        return max(0, len(self.data) - self.seq_len)
    
    def __getitem__(self, idx):
        chunk = self.data[idx:idx + self.seq_len + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


def get_dataloaders(config):
    """Load WikiText-2 dataset."""
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1") # change dataset if needed
    
    def tokenize(text):
        return [min(ord(c), config.vocab_size - 1) for c in text]
    
    train_text = ' '.join(dataset['train']['text'])
    val_text = ' '.join(dataset['validation']['text'])
    
    train_data = tokenize(train_text)
    val_data = tokenize(val_text)
    
    train_dataset = TextDataset(train_data, config.max_seq_len, config.vocab_size)
    val_dataset = TextDataset(val_data, config.max_seq_len, config.vocab_size)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    
    return train_loader, val_loader


@torch.no_grad()
def evaluate(model, val_loader, config):
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    for batch_idx, (x, y) in enumerate(val_loader):
        if batch_idx >= config.eval_iters:
            break
        
        x, y = x.to(config.device), y.to(config.device)
        _, loss = model(x, y)
        total_loss += loss.item()
        num_batches += 1
    
    avg_loss = total_loss / max(num_batches, 1)
    perplexity = math.exp(min(avg_loss, 20))
    
    return avg_loss, perplexity


def train_epoch(model, train_loader, optimizer, config, epoch, use_manifold_projection):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for x, y in pbar:
        x, y = x.to(config.device), y.to(config.device)
        
        optimizer.zero_grad()
        _, loss = model(x, y)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        if use_manifold_projection:
            model.project_to_manifold()
        
        total_loss += loss.item()
        num_batches += 1
        
        pbar.set_postfix({'loss': total_loss / num_batches})
    
    return total_loss / num_batches


def train_manifold_transformer(config):
    """
    Train transformer with manifold constraints.
    
    Returns:
        model: trained model
        metrics_history: list of dicts with metrics per epoch
    """
    print(f"\n{'='*60}")
    print(f"Training: {config.get_name()}")
    print(f"  FFN constrained: {config.constraint_ffn}")
    print(f"  Attention constrained: {config.constraint_attention}")
    print(f"  Manifold: {config.manifold_type}")
    print(f"{'='*60}\n")
    
    if torch.cuda.is_available():
        config.device = 'cuda'
    elif torch.backends.mps.is_available():
        config.device = 'mps'
    else:
        config.device = 'cpu'
    
    print(f"Using device: {config.device}")
    
    torch.manual_seed(config.seed)
    
    wandb.init(
        project=config.wandb_project,
        entity=config.wandb_entity,
        name=config.get_name(),
        config={
            'constraint_ffn': config.constraint_ffn,
            'constraint_attention': config.constraint_attention,
            'manifold_type': config.manifold_type,
            'd_model': config.d_model,
            'n_layers': config.n_layers,
            'n_heads': config.n_heads,
            'batch_size': config.batch_size,
            'learning_rate': config.learning_rate,
        }
    )
    
    print("Loading data...")
    train_loader, val_loader = get_dataloaders(config)
    
    print("Creating model...")
    model = ManifoldTransformer(config).to(config.device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    optimizer_type = config.get_optimizer_type()
    print(f"Using optimizer: {optimizer_type}")
    
    use_manifold_projection = False
    
    if optimizer_type == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    elif optimizer_type == 'manifold_muon':
        optimizer = ManifoldMuon(model.parameters(), lr=config.muon_lr)
    elif optimizer_type == 'sphere_optimizer':
        optimizer = SphereOptimizer(model.parameters(), lr=config.muon_lr)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")
    

    if optimizer_type == 'adam' and (config.constraint_ffn or config.constraint_attention):
        use_manifold_projection = True
    
    metrics_history = []
    best_val_loss = float('inf')
    
    print("\nStarting training...")
    for epoch in range(1, config.num_epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, config, epoch, use_manifold_projection)
        val_loss, perplexity = evaluate(model, val_loader, config)
        
        cond_stats = model.get_condition_number_stats()
        
        metrics = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'perplexity': perplexity,
            'cond_mean': cond_stats['mean'],
            'cond_max': cond_stats['max'],
            'cond_min': cond_stats['min'],
            'cond_std': cond_stats['std'],
        }
        metrics_history.append(metrics)
        
        wandb.log({
            'epoch': epoch,
            'train/loss': train_loss,
            'val/loss': val_loss,
            'val/perplexity': perplexity,
            'condition_number/mean': cond_stats['mean'],
            'condition_number/max': cond_stats['max'],
            'condition_number/min': cond_stats['min'],
            'condition_number/std': cond_stats['std'],
        })
        
        for layer_name, cond in cond_stats['by_layer'].items():
            wandb.log({f'condition_number/{layer_name}': cond})
        
        print(f"\nEpoch {epoch}:")
        print(f"  Train loss: {train_loss:.4f}")
        print(f"  Val loss: {val_loss:.4f}")
        print(f"  Perplexity: {perplexity:.2f}")
        print(f"  Condition number: mean={cond_stats['mean']:.2f}, max={cond_stats['max']:.2f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"{config.get_name()}_best.pt")
    
    print(f"\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    
    wandb.finish()
    
    return model, metrics_history