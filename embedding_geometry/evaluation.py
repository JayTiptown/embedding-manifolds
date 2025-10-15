import torch
import numpy as np
from scipy.stats import spearmanr


def evaluate_word_similarity(model, wordsim_dataset, vocab):
    """
    Evaluate embeddings on word similarity task.
    
    Returns Spearman correlation between predicted similarities
    and human judgments.
    """
    predictions = []
    ground_truth = []
    
    valid_pairs = 0
    total_pairs = len(wordsim_dataset.pairs)
    
    for word1, word2, score in wordsim_dataset.pairs:
        idx1 = vocab.get_idx(word1)
        idx2 = vocab.get_idx(word2)
        
        if idx1 == 0 or idx2 == 0:
            continue
        
        with torch.no_grad():
            sim = model.similarity(idx1, idx2).item()
        
        predictions.append(sim)
        ground_truth.append(score)
        valid_pairs += 1
    
    if len(predictions) < 2:
        return 0.0, 0
    
    correlation, pvalue = spearmanr(predictions, ground_truth)
    
    return correlation, valid_pairs


def evaluate_analogies(model, analogy_dataset, vocab, k=5):
    """
    Evaluate embeddings on analogy task.
    
    For each analogy (a, b, c, d), compute d' = b - a + c
    and check if d is in top-k nearest neighbors.
    """
    correct = 0
    total = 0
    
    category_stats = {}
    
    for analogy in analogy_dataset.analogies:
        a_idx = vocab.get_idx(analogy['a'])
        b_idx = vocab.get_idx(analogy['b'])
        c_idx = vocab.get_idx(analogy['c'])
        d_idx = vocab.get_idx(analogy['d'])
        
        if 0 in [a_idx, b_idx, c_idx, d_idx]:
            continue
        
        with torch.no_grad():
            top_k_indices, _ = model.analogy(a_idx, b_idx, c_idx, k=k)
        
        category = analogy['category']
        if category not in category_stats:
            category_stats[category] = {'correct': 0, 'total': 0}
        
        if d_idx in top_k_indices:
            correct += 1
            category_stats[category]['correct'] += 1
        
        total += 1
        category_stats[category]['total'] += 1
    
    accuracy = correct / total if total > 0 else 0.0
    
    category_accuracies = {}
    for cat, stats in category_stats.items():
        category_accuracies[cat] = stats['correct'] / stats['total'] if stats['total'] > 0 else 0.0
    
    return accuracy, total, category_accuracies


def compute_embedding_statistics(model):
    """Compute statistics about the embeddings."""
    embeddings = model.embeddings.data
    
    norms = torch.norm(embeddings, dim=1)
    mean_norm = norms.mean().item()
    std_norm = norms.std().item()
    min_norm = norms.min().item()
    max_norm = norms.max().item()
    
    distances = torch.cdist(embeddings, embeddings)
    distances.fill_diagonal_(float('inf'))
    nearest_distances = distances.min(dim=1)[0]
    mean_nearest_distance = nearest_distances.mean().item()
    
    if embeddings.size(0) > 1:
        U, S, V = torch.linalg.svd(embeddings, full_matrices=False)
        explained_variance = (S ** 2) / (S ** 2).sum()
        intrinsic_dim_90 = (explained_variance.cumsum(dim=0) < 0.9).sum().item() + 1
        intrinsic_dim_95 = (explained_variance.cumsum(dim=0) < 0.95).sum().item() + 1
        condition_number = (S.max() / (S.min() + 1e-8)).item()
    else:
        intrinsic_dim_90 = 0
        intrinsic_dim_95 = 0
        condition_number = 0
    
    stats = {
        'mean_norm': mean_norm,
        'std_norm': std_norm,
        'min_norm': min_norm,
        'max_norm': max_norm,
        'mean_nearest_distance': mean_nearest_distance,
        'intrinsic_dim_90': intrinsic_dim_90,
        'intrinsic_dim_95': intrinsic_dim_95,
        'condition_number': condition_number,
    }
    
    return stats


def get_nearest_neighbors(model, word_idx, vocab, k=10):
    """Get k nearest neighbors for a word."""
    with torch.no_grad():
        word_vec = model.embeddings[word_idx]
        
        if model.manifold == 'sphere':
            scores = torch.matmul(model.embeddings, word_vec)
        else:
            scores = torch.matmul(model.embeddings, word_vec)
            norms = torch.norm(model.embeddings, dim=1)
            scores = scores / (norms * torch.norm(word_vec) + 1e-8)
        
        scores[word_idx] = -float('inf')
        
        top_k = torch.topk(scores, k)
        indices = top_k.indices.cpu().numpy()
        scores = top_k.values.cpu().numpy()
        
        neighbors = []
        for idx, score in zip(indices, scores):
            if idx in vocab.idx2word:
                neighbors.append((vocab.idx2word[idx], score))
        
        return neighbors

