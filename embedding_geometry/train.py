import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import numpy as np

from .data_loader import Vocabulary, SkipGramDataset, load_text_corpus, WordSimDataset, GoogleAnalogyDataset
from .models import WordEmbeddings, NegativeSamplingLoss
from .evaluation import evaluate_word_similarity, evaluate_analogies, compute_embedding_statistics, get_nearest_neighbors
from .visualization import log_embeddings_to_wandb, visualize_analogy, create_manifold_geometry_plot
from optimizers import ManifoldMuon, SphereOptimizer


def train_epoch(model, dataloader, criterion, optimizer, epoch, device, log_interval=100, use_manifold_optimizer=False):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, (center_ids, context_ids) in enumerate(pbar):
        center_ids = center_ids.to(device)
        context_ids = context_ids.to(device)
        
        optimizer.zero_grad()
        
        loss = criterion(model, center_ids, context_ids, model.vocab_size)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        if not use_manifold_optimizer:
            model.project_to_manifold()
        
        total_loss += loss.item()
        num_batches += 1
        
        if batch_idx % log_interval == 0:
            avg_loss = total_loss / num_batches
            pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
    
    return total_loss / num_batches


def train_embeddings(
    manifold='euclidean',
    embedding_dim=100,
    num_epochs=10,
    batch_size=512,
    learning_rate=0.001,
    num_negatives=5,
    window_size=5,
    min_freq=5,
    max_corpus_chars=1000000,
    eval_interval=1,
    device=None,
    seed=42
):
    """
    Train word embeddings on a manifold.
    
    Args:
        manifold: 'euclidean', 'sphere', or 'stiefel'
        embedding_dim: dimension of embeddings
        num_epochs: number of training epochs
        batch_size: training batch size
        learning_rate: learning rate
        num_negatives: number of negative samples
        window_size: context window size
        min_freq: minimum word frequency
        max_corpus_chars: maximum corpus size
        eval_interval: evaluate every N epochs
        device: device to train on
        seed: random seed
    """
    if device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    
    print(f"Using device: {device}")
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    run_name = f"{manifold}_d{embedding_dim}_e{num_epochs}"
    
    wandb.init(
        project="embedding-manifolds",
        name=run_name,
        config={
            'manifold': manifold,
            'embedding_dim': embedding_dim,
            'num_epochs': num_epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'num_negatives': num_negatives,
            'window_size': window_size,
            'min_freq': min_freq,
        }
    )
    
    print(f"Loading text corpus...")
    text = load_text_corpus(max_chars=max_corpus_chars)
    
    print(f"Building vocabulary...")
    vocab = Vocabulary(min_freq=min_freq)
    vocab.build([text])
    print(f"Vocabulary size: {len(vocab)}")
    
    print(f"Creating skip-gram dataset...")
    dataset = SkipGramDataset(text, vocab, window_size=window_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    print(f"Dataset size: {len(dataset)} pairs")
    
    print(f"Loading evaluation datasets...")
    wordsim_dataset = WordSimDataset()
    analogy_dataset = GoogleAnalogyDataset()
    
    print(f"Initializing model on {manifold} manifold...")
    model = WordEmbeddings(len(vocab), embedding_dim, manifold=manifold)
    model = model.to(device)
    
    criterion = NegativeSamplingLoss(num_negatives=num_negatives)
    
    use_manifold_optimizer = manifold in ['stiefel', 'sphere']
    if manifold == 'stiefel':
        optimizer = ManifoldMuon(model.parameters(), lr=learning_rate)
    elif manifold == 'sphere':
        optimizer = SphereOptimizer(model.parameters(), lr=learning_rate)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    sample_words = ['king', 'queen', 'man', 'woman', 'computer', 'science', 
                   'dog', 'cat', 'car', 'truck', 'happy', 'sad']
    
    print(f"\nStarting training...")
    for epoch in range(1, num_epochs + 1):
        train_loss = train_epoch(model, dataloader, criterion, optimizer, epoch, device, 
                                use_manifold_optimizer=use_manifold_optimizer)
        
        print(f"\nEpoch {epoch}/{num_epochs} - Train Loss: {train_loss:.4f}")
        
        wandb.log({
            'epoch': epoch,
            'train/loss': train_loss,
        })
        
        if epoch % eval_interval == 0 or epoch == num_epochs:
            print(f"\nEvaluating...")
            
            model.eval()
            with torch.no_grad():
                wordsim_corr, wordsim_pairs = evaluate_word_similarity(model, wordsim_dataset, vocab)
                analogy_acc, analogy_total, category_accs = evaluate_analogies(model, analogy_dataset, vocab, k=5)
                
                embedding_stats = compute_embedding_statistics(model)
                
                print(f"WordSim-353: ρ={wordsim_corr:.4f} ({wordsim_pairs} pairs)")
                print(f"Analogies: Acc={analogy_acc:.4f} ({analogy_total} questions)")
                print(f"Embedding stats: mean_norm={embedding_stats['mean_norm']:.4f}, "
                      f"intrinsic_dim_90={embedding_stats['intrinsic_dim_90']}")
                
                for cat, acc in category_accs.items():
                    print(f"  {cat}: {acc:.4f}")
                
                wandb.log({
                    'epoch': epoch,
                    'eval/wordsim_correlation': wordsim_corr,
                    'eval/wordsim_pairs': wordsim_pairs,
                    'eval/analogy_accuracy': analogy_acc,
                    'eval/analogy_total': analogy_total,
                    'stats/mean_norm': embedding_stats['mean_norm'],
                    'stats/std_norm': embedding_stats['std_norm'],
                    'stats/min_norm': embedding_stats['min_norm'],
                    'stats/max_norm': embedding_stats['max_norm'],
                    'stats/mean_nearest_distance': embedding_stats['mean_nearest_distance'],
                    'stats/intrinsic_dim_90': embedding_stats['intrinsic_dim_90'],
                    'stats/intrinsic_dim_95': embedding_stats['intrinsic_dim_95'],
                    'stats/condition_number': embedding_stats['condition_number'],
                })
                
                for cat, acc in category_accs.items():
                    wandb.log({f'eval/analogy_{cat}': acc})
                
                print(f"\nExample nearest neighbors:")
                example_words = ['king', 'computer', 'happy']
                for word in example_words:
                    idx = vocab.get_idx(word)
                    if idx != 0:
                        neighbors = get_nearest_neighbors(model, idx, vocab, k=5)
                        neighbor_str = ', '.join([f"{w}({s:.3f})" for w, s in neighbors])
                        print(f"  {word}: {neighbor_str}")
                
                print(f"\nVisualizing embeddings...")
                log_embeddings_to_wandb(model, vocab, epoch, sample_words=sample_words, n_words=100)
                
                if embedding_dim >= 3:
                    geometry_fig = create_manifold_geometry_plot(model, vocab, n_points=200)
                    wandb.log({
                        'embeddings/manifold_geometry': wandb.Plotly(geometry_fig),
                        'epoch': epoch
                    })
                
                example_analogies = [
                    ('king', 'queen', 'man'),
                    ('paris', 'france', 'london'),
                ]
                for a, b, c in example_analogies:
                    a_idx = vocab.get_idx(a)
                    b_idx = vocab.get_idx(b)
                    c_idx = vocab.get_idx(c)
                    
                    if all(idx != 0 for idx in [a_idx, b_idx, c_idx]):
                        top_k_indices, top_k_scores = model.analogy(a_idx, b_idx, c_idx, k=5)
                        predicted_words = [vocab.idx2word.get(idx, '<UNK>') for idx in top_k_indices]
                        
                        print(f"\n  Analogy: {a}:{b} :: {c}:?")
                        for word, score in zip(predicted_words, top_k_scores):
                            print(f"    {word} ({score:.3f})")
                        
                        analogy_fig = visualize_analogy(model, vocab, a, b, c, predicted_words[:3])
                        wandb.log({
                            f'analogies/{a}_{b}_{c}': wandb.Plotly(analogy_fig),
                            'epoch': epoch
                        })
    
    print(f"\nTraining complete!")
    
    checkpoint_path = f"{run_name}.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab': vocab,
        'config': {
            'manifold': manifold,
            'embedding_dim': embedding_dim,
            'vocab_size': len(vocab),
        }
    }, checkpoint_path)
    print(f"Model saved to {checkpoint_path}")
    
    wandb.finish()
    
    return model, vocab

