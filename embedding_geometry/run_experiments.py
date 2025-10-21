#!/usr/bin/env python3
"""
Run Phase 1 experiments: Word Embeddings on Manifolds

This script trains word embeddings on different manifolds (Euclidean, Sphere, Stiefel)
and evaluates them on WordSim-353 and Google Analogies benchmarks.
"""

import argparse
import torch
import wandb
from .train import train_embeddings


def run_all_manifolds(embedding_dim=100, num_epochs=10, device=None):
    """Run experiments for all manifold types."""
    
    if device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    
    print(f"Using device: {device}")
    
    manifolds = ['euclidean', 'sphere', 'stiefel']
    results = {}
    
    for manifold in manifolds:
        print(f"\n{'='*60}")
        print(f"Training embeddings on {manifold.upper()} manifold")
        print(f"{'='*60}\n")
        
        model, vocab = train_embeddings(
            manifold=manifold,
            embedding_dim=embedding_dim,
            num_epochs=num_epochs,
            batch_size=512,
            learning_rate=0.001,
            num_negatives=5,
            window_size=5,
            min_freq=5,
            max_corpus_chars=2000000,
            eval_interval=2,
            device=device,
            seed=42
        )
        
        results[manifold] = {
            'model': model,
            'vocab': vocab
        }
    
    print(f"\n{'='*60}")
    print(f"All experiments complete!")
    print(f"{'='*60}\n")
    
    return results


def run_single_experiment(
    manifold='euclidean',
    embedding_dim=100,
    num_epochs=10,
    batch_size=512,
    learning_rate=0.001,
    num_negatives=5,
    window_size=5,
    min_freq=5,
    max_corpus_chars=2000000,
    eval_interval=2,
    device=None,
    seed=42,
    use_geodesic=False
):
    """Run a single experiment with specified parameters."""
    
    if device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    
    print(f"\n{'='*60}")
    print(f"Training {embedding_dim}D embeddings on {manifold.upper()} manifold")
    print(f"{'='*60}\n")
    
    model, vocab = train_embeddings(
        manifold=manifold,
        embedding_dim=embedding_dim,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_negatives=num_negatives,
        window_size=window_size,
        min_freq=min_freq,
        max_corpus_chars=max_corpus_chars,
        eval_interval=eval_interval,
        device=device,
        seed=seed,
        use_geodesic=use_geodesic
    )
    
    return model, vocab


def main():
    parser = argparse.ArgumentParser(description='Train word embeddings on manifolds')
    
    parser.add_argument('--manifold', type=str, default='all',
                       choices=['all', 'euclidean', 'sphere', 'stiefel'],
                       help='Manifold type to use (default: all)')
    
    parser.add_argument('--embedding-dim', type=int, default=100,
                       help='Embedding dimension (default: 100)')
    
    parser.add_argument('--num-epochs', type=int, default=10,
                       help='Number of training epochs (default: 10)')
    
    parser.add_argument('--batch-size', type=int, default=512,
                       help='Batch size (default: 512)')
    
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')
    
    parser.add_argument('--num-negatives', type=int, default=5,
                       help='Number of negative samples (default: 5)')
    
    parser.add_argument('--window-size', type=int, default=5,
                       help='Context window size (default: 5)')
    
    parser.add_argument('--min-freq', type=int, default=5,
                       help='Minimum word frequency (default: 5)')
    
    parser.add_argument('--max-corpus-chars', type=int, default=2000000,
                       help='Maximum corpus size in characters (default: 2000000)')
    
    parser.add_argument('--eval-interval', type=int, default=2,
                       help='Evaluate every N epochs (default: 2)')
    
    parser.add_argument('--device', type=str, default=None,
                       choices=['cpu', 'cuda', 'mps'],
                       help='Device to use (default: auto-detect)')
    
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    
    parser.add_argument('--use-geodesic', action='store_true',
                       help='Use exponential map (true geodesic) for sphere manifold')
    
    args = parser.parse_args()
    
    if args.manifold == 'all':
        results = run_all_manifolds(
            embedding_dim=args.embedding_dim,
            num_epochs=args.num_epochs,
            device=args.device
        )
    else:
        model, vocab = run_single_experiment(
            manifold=args.manifold,
            embedding_dim=args.embedding_dim,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            num_negatives=args.num_negatives,
            window_size=args.window_size,
            min_freq=args.min_freq,
            max_corpus_chars=args.max_corpus_chars,
            eval_interval=args.eval_interval,
            device=args.device,
            seed=args.seed,
            use_geodesic=args.use_geodesic
        )


if __name__ == '__main__':
    main()

