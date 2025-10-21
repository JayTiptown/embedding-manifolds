"""
Run all experiments and compare results.
"""

import json
import torch
from .config import get_experiment_configs
from .train import train_manifold_transformer


def run_all_experiments():
    """
    Run all experiments: baseline, FFN-only, attention-only, full.
    
    Returns:
        dict: {experiment_name: metrics_history}
    """
    configs = get_experiment_configs()
    all_results = {}
    
    for config in configs:
        name = config.get_name()
        
        print(f"\n{'#'*80}")
        print(f"# Experiment: {name}")
        print(f"{'#'*80}\n")
        
        model, metrics = train_manifold_transformer(config)
        all_results[name] = metrics
        
        # Save results
        with open(f"{name}_metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)
    
    # Save all results
    with open("all_experiments.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Print summary
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    
    for name, metrics in all_results.items():
        final = metrics[-1]
        print(f"\n{name}:")
        print(f"  Final val loss: {final['val_loss']:.4f}")
        print(f"  Final perplexity: {final['perplexity']:.2f}")
        print(f"  Mean condition number: {final['cond_mean']:.2f}")
        print(f"  Max condition number: {final['cond_max']:.2f}")
    
    return all_results


if __name__ == '__main__':
    run_all_experiments()