import torch
import json
from config import Config
from models import SmallTransformer
from train import get_dataloaders, train_model
from optimizers import ManifoldMuon, StandardMuon


def run_experiment(config, constrained, optimizer_type):
    """Run a single experiment configuration."""
    torch.manual_seed(config.seed)
    
    model = SmallTransformer(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        d_ff=config.d_ff,
        max_seq_len=config.max_seq_len,
        dropout=config.dropout,
        constrained=constrained
    ).to(config.device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    if optimizer_type == 'adam':
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    elif optimizer_type == 'muon':
        optimizer = StandardMuon(model.parameters(), lr=config.muon_lr)
    elif optimizer_type == 'manifold_muon':
        optimizer = ManifoldMuon(model.parameters(), lr=config.muon_lr)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")
    
    train_loader, val_loader = get_dataloaders(config)
    
    constraint_str = "constrained" if constrained else "unconstrained"
    run_name = f"{constraint_str}_{optimizer_type}"
    
    run_config = {
        'constrained': constrained,
        'optimizer': optimizer_type,
        'd_model': config.d_model,
        'n_layers': config.n_layers,
        'n_heads': config.n_heads,
        'd_ff': config.d_ff,
        'vocab_size': config.vocab_size,
        'max_seq_len': config.max_seq_len,
        'batch_size': config.batch_size,
        'num_epochs': config.num_epochs,
        'lr': config.lr if optimizer_type == 'adam' else config.muon_lr,
        'total_params': total_params,
    }
    
    metrics_history = train_model(model, optimizer, train_loader, val_loader, config, run_name, run_config)
    
    with open(f"{run_name}_metrics.json", 'w') as f:
        json.dump(metrics_history, f, indent=2)
    
    return metrics_history


def main():
    config = Config()
    
    experiments = [
        ("unconstrained", "adam"),
        ("unconstrained", "muon"),
        ("constrained", "adam"),
        ("constrained", "manifold_muon"),
    ]
    
    all_results = {}
    
    for constrained_str, optimizer_type in experiments:
        constrained = (constrained_str == "constrained")
        
        print(f"\n{'#'*60}")
        print(f"Experiment: {constrained_str.upper()} + {optimizer_type.upper()}")
        print(f"{'#'*60}\n")
        
        metrics = run_experiment(config, constrained, optimizer_type)
        all_results[f"{constrained_str}_{optimizer_type}"] = metrics
    
    with open("all_experiments_results.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\n" + "="*60)
    print("All experiments completed!")
    print("="*60)
    
    print("\nFinal Results Summary:")
    for name, metrics in all_results.items():
        final_metrics = metrics[-1]
        print(f"\n{name}:")
        print(f"  Final val_loss: {final_metrics['val_loss']:.4f}")
        print(f"  Final perplexity: {final_metrics['perplexity']:.2f}")
        print(f"  Avg condition number: {final_metrics['avg_condition_number']:.2f}")
        print(f"  Max condition number: {final_metrics['max_condition_number']:.2f}")
