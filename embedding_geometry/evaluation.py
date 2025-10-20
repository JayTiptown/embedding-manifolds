import torch
import numpy as np
from scipy.stats import spearmanr
from collections import defaultdict


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


def compute_cooccurrence_matrix(dataset, vocab_size):
    """
    Compute co-occurrence matrix from skip-gram dataset.
    Uses the training pairs to build a reference space.
    """
    cooccurrence = defaultdict(lambda: defaultdict(int))
    
    for center, context in dataset.pairs:
        cooccurrence[center][context] += 1
        cooccurrence[context][center] += 1
    
    return cooccurrence


def compute_precision_at_k(model, dataset, vocab, k_values=[5, 10, 20]):
    """
    Compute Precision@k: overlap between embedding neighbors and co-occurrence neighbors.
    Measures how well embedding space preserves local neighborhood structure.
    """
    cooccurrence = compute_cooccurrence_matrix(dataset, len(vocab))
    
    precisions = {k: [] for k in k_values}
    max_k = max(k_values)
    
    sample_size = min(500, len(vocab) - 1)
    sample_indices = np.random.choice(range(1, len(vocab)), size=sample_size, replace=False)
    
    for word_idx in sample_indices:
        if word_idx not in cooccurrence or len(cooccurrence[word_idx]) < 5:
            continue
        
        cooccur_neighbors = sorted(cooccurrence[word_idx].items(), 
                                   key=lambda x: x[1], reverse=True)[:max_k]
        cooccur_set = {idx for idx, _ in cooccur_neighbors}
        
        with torch.no_grad():
            word_vec = model.embeddings[word_idx]
            
            if model.manifold == 'sphere':
                scores = torch.matmul(model.embeddings, word_vec)
            else:
                scores = torch.matmul(model.embeddings, word_vec)
                norms = torch.norm(model.embeddings, dim=1)
                scores = scores / (norms * torch.norm(word_vec) + 1e-8)
            
            scores[word_idx] = -float('inf')
            
            top_k = torch.topk(scores, max_k)
            embedding_neighbors = top_k.indices.cpu().numpy()
        
        for k in k_values:
            embedding_set = set(embedding_neighbors[:k])
            cooccur_set_k = {idx for idx, _ in cooccur_neighbors[:k]}
            
            if len(cooccur_set_k) > 0:
                precision = len(embedding_set & cooccur_set_k) / k
                precisions[k].append(precision)
    
    avg_precisions = {k: np.mean(vals) if vals else 0.0 for k, vals in precisions.items()}
    
    return avg_precisions


def compute_rank_distortion(model, dataset, vocab, sample_size=500):
    """
    Compute mean rank distortion: how much do neighbor rankings change
    between co-occurrence space and embedding space.
    """
    cooccurrence = compute_cooccurrence_matrix(dataset, len(vocab))
    
    rank_distortions = []
    
    sample_indices = np.random.choice(range(1, len(vocab)), 
                                     size=min(sample_size, len(vocab) - 1), 
                                     replace=False)
    
    for word_idx in sample_indices:
        if word_idx not in cooccurrence or len(cooccurrence[word_idx]) < 10:
            continue
        
        cooccur_items = list(cooccurrence[word_idx].items())
        cooccur_indices = [idx for idx, _ in cooccur_items]
        
        if len(cooccur_indices) < 10:
            continue
        
        cooccur_ranks = {idx: rank for rank, (idx, _) in 
                        enumerate(sorted(cooccur_items, key=lambda x: x[1], reverse=True))}
        
        with torch.no_grad():
            word_vec = model.embeddings[word_idx]
            
            if model.manifold == 'sphere':
                scores = torch.matmul(model.embeddings, word_vec)
            else:
                scores = torch.matmul(model.embeddings, word_vec)
                norms = torch.norm(model.embeddings, dim=1)
                scores = scores / (norms * torch.norm(word_vec) + 1e-8)
            
            embedding_scores = {idx: scores[idx].item() for idx in cooccur_indices}
            embedding_ranks = {idx: rank for rank, (idx, _) in 
                             enumerate(sorted(embedding_scores.items(), 
                                            key=lambda x: x[1], reverse=True))}
        
        rank_diffs = [abs(cooccur_ranks[idx] - embedding_ranks[idx]) 
                     for idx in cooccur_indices]
        
        if rank_diffs:
            rank_distortions.append(np.mean(rank_diffs))
    
    return np.mean(rank_distortions) if rank_distortions else 0.0


def compute_isotropy_score(model, sample_size=1000):
    """
    Compute isotropy score: measures how uniformly embeddings are distributed.
    Lower variance in cosine similarities indicates better isotropy.
    Returns normalized score where higher is better (more isotropic).
    """
    embeddings = model.embeddings.data
    n_embeddings = embeddings.size(0)
    
    sample_size = min(sample_size, n_embeddings)
    sample_indices = torch.randperm(n_embeddings)[:sample_size]
    sample_embeddings = embeddings[sample_indices]
    
    normalized = sample_embeddings / (torch.norm(sample_embeddings, dim=1, keepdim=True) + 1e-8)
    
    cosine_similarities = torch.matmul(normalized, normalized.t())
    
    mask = ~torch.eye(sample_size, dtype=torch.bool, device=embeddings.device)
    similarities = cosine_similarities[mask]
    
    similarity_variance = torch.var(similarities).item()
    similarity_mean = torch.mean(similarities).item()
    
    isotropy_score = 1.0 / (1.0 + similarity_variance)
    
    return {
        'isotropy_score': isotropy_score,
        'similarity_variance': similarity_variance,
        'similarity_mean': similarity_mean,
    }


def compute_triangle_inequality_violations(model, sample_size=500):
    """
    Check triangle inequality: d(a,c) <= d(a,b) + d(b,c).
    Returns the fraction of violations and mean violation magnitude.
    """
    embeddings = model.embeddings.data
    n_embeddings = embeddings.size(0)
    
    sample_size = min(sample_size, n_embeddings)
    sample_indices = torch.randperm(n_embeddings)[:sample_size]
    sample_embeddings = embeddings[sample_indices]
    
    distances = torch.cdist(sample_embeddings, sample_embeddings)
    
    violations = 0
    total_checks = 0
    violation_magnitudes = []
    
    num_triplets = min(1000, sample_size * (sample_size - 1) * (sample_size - 2) // 6)
    
    for _ in range(num_triplets):
        i, j, k = torch.randint(0, sample_size, (3,))
        if i == j or j == k or i == k:
            continue
        
        d_ik = distances[i, k]
        d_ij = distances[i, j]
        d_jk = distances[j, k]
        
        violation = d_ik - (d_ij + d_jk)
        
        if violation > 1e-6:
            violations += 1
            violation_magnitudes.append(violation.item())
        
        total_checks += 1
    
    violation_rate = violations / total_checks if total_checks > 0 else 0.0
    mean_violation = np.mean(violation_magnitudes) if violation_magnitudes else 0.0
    
    return {
        'violation_rate': violation_rate,
        'mean_violation': mean_violation,
        'total_checks': total_checks,
    }


def compute_participation_ratio(model):
    """
    Compute participation ratio: effective dimensionality measure.
    PR = (sum of eigenvalues)^2 / sum of squared eigenvalues.
    Higher values indicate more dimensions are being used effectively.
    """
    embeddings = model.embeddings.data
    
    if embeddings.size(0) <= 1:
        return 0.0
    
    centered = embeddings - embeddings.mean(dim=0, keepdim=True)
    
    U, S, V = torch.linalg.svd(centered, full_matrices=False)
    
    eigenvalues = S ** 2
    
    sum_eig = eigenvalues.sum()
    sum_sq_eig = (eigenvalues ** 2).sum()
    
    participation_ratio = (sum_eig ** 2) / (sum_sq_eig + 1e-8)
    
    normalized_pr = participation_ratio.item() / min(embeddings.size(0), embeddings.size(1))
    
    return {
        'participation_ratio': participation_ratio.item(),
        'normalized_pr': normalized_pr,
    }


def compute_distortion_metrics(model, dataset, vocab):
    """
    Compute all distortion metrics: Precision@k and rank distortion.
    """
    precision_at_k = compute_precision_at_k(model, dataset, vocab, k_values=[5, 10, 20])
    rank_distortion = compute_rank_distortion(model, dataset, vocab, sample_size=500)
    
    return {
        'precision_at_5': precision_at_k[5],
        'precision_at_10': precision_at_k[10],
        'precision_at_20': precision_at_k[20],
        'rank_distortion': rank_distortion,
    }


def compute_geometric_quality_metrics(model):
    """
    Compute all geometric quality metrics: isotropy and triangle inequality.
    """
    isotropy = compute_isotropy_score(model, sample_size=1000)
    triangle = compute_triangle_inequality_violations(model, sample_size=500)
    
    return {
        'isotropy_score': isotropy['isotropy_score'],
        'similarity_variance': isotropy['similarity_variance'],
        'similarity_mean': isotropy['similarity_mean'],
        'triangle_violation_rate': triangle['violation_rate'],
        'triangle_mean_violation': triangle['mean_violation'],
    }


def compute_capacity_metrics(model):
    """
    Compute capacity indicators: participation ratio.
    """
    pr = compute_participation_ratio(model)
    
    return {
        'participation_ratio': pr['participation_ratio'],
        'normalized_participation_ratio': pr['normalized_pr'],
    }

