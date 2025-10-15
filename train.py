import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from tqdm import tqdm
import math
import wandb
from config import Config
from models import SmallTransformer
from manifolds import stiefel_project, sphere_project


class TextDataset(Dataset):
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
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    
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


def project_manifold_params(model):
    """Project constrained parameters back to their manifolds."""
    if not model.constrained:
        return
    
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'embedding.weight' in name and param.requires_grad:
                param.copy_(sphere_project(param))
            elif 'weight' in name and param.dim() >= 2 and param.requires_grad:
                if param.shape[0] <= param.shape[1]:
                    param.copy_(stiefel_project(param))


@torch.no_grad()
def evaluate(model, val_loader, config):
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


def train_epoch(model, train_loader, optimizer, config, epoch):
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
        
        if model.constrained:
            project_manifold_params(model)
        
        total_loss += loss.item()
        num_batches += 1
        
        pbar.set_postfix({'loss': total_loss / num_batches})
    
    return total_loss / num_batches


def train_model(model, optimizer, train_loader, val_loader, config, run_name, run_config=None):
    print(f"\n{'='*50}")
    print(f"Training: {run_name}")
    print(f"{'='*50}")
    
    wandb.init(
        project=config.wandb_project,
        entity=config.wandb_entity,
        name=run_name,
        config=run_config or {},
    )
    
    best_val_loss = float('inf')
    metrics_history = []
    
    for epoch in range(config.num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, config, epoch + 1)
        val_loss, perplexity = evaluate(model, val_loader, config)
        
        cond_numbers = model.get_condition_numbers()
        avg_cond = sum([c for _, c in cond_numbers]) / max(len(cond_numbers), 1)
        max_cond = max([c for _, c in cond_numbers]) if cond_numbers else 0
        min_cond = min([c for _, c in cond_numbers]) if cond_numbers else 0
        
        metrics = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'perplexity': perplexity,
            'avg_condition_number': avg_cond,
            'max_condition_number': max_cond
        }
        metrics_history.append(metrics)
        
        wandb.log({
            'epoch': epoch + 1,
            'train/loss': train_loss,
            'val/loss': val_loss,
            'val/perplexity': perplexity,
            'condition_number/avg': avg_cond,
            'condition_number/max': max_cond,
            'condition_number/min': min_cond,
        })
        
        cond_table_data = [[name, cond] for name, cond in cond_numbers]
        if cond_table_data:
            wandb.log({
                'condition_numbers': wandb.Table(
                    columns=['layer', 'condition_number'],
                    data=cond_table_data
                )
            })
        
        print(f"Epoch {epoch + 1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
              f"perplexity={perplexity:.2f}, avg_cond={avg_cond:.2f}, max_cond={max_cond:.2f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"{run_name}_best.pt")
            wandb.run.summary["best_val_loss"] = best_val_loss
    
    wandb.finish()
    
    return metrics_history

