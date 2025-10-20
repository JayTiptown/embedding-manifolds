"""
Embedding Geometry - Phase 1: Word Embeddings on Manifolds

This package implements word embeddings with different geometric constraints:
- Euclidean: Standard unconstrained embeddings
- Sphere: Embeddings on the unit hypersphere
- Stiefel: Embeddings on the Stiefel manifold (orthogonal rows)

The package includes:
- Data loading for WordSim-353 and Google Analogies benchmarks
- Skip-gram training with negative sampling
- Comprehensive evaluation metrics:
  * Performance: similarity correlation, analogy accuracy
  * Distortion: Precision@k, rank distortion
  * Geometric quality: isotropy, triangle inequality
  * Capacity: participation ratio, intrinsic dimensionality
- 2D/3D visualizations with wandb
"""

from .data_loader import (
    Vocabulary,
    WordSimDataset,
    GoogleAnalogyDataset,
    SkipGramDataset,
    load_text_corpus
)

from .models import (
    WordEmbeddings,
    NegativeSamplingLoss
)

from .evaluation import (
    evaluate_word_similarity,
    evaluate_analogies,
    compute_embedding_statistics,
    get_nearest_neighbors,
    compute_distortion_metrics,
    compute_geometric_quality_metrics,
    compute_capacity_metrics
)

from .visualization import (
    visualize_embeddings,
    log_embeddings_to_wandb,
    visualize_analogy,
    create_manifold_geometry_plot
)

from .train import train_embeddings

__all__ = [
    'Vocabulary',
    'WordSimDataset',
    'GoogleAnalogyDataset',
    'SkipGramDataset',
    'load_text_corpus',
    'WordEmbeddings',
    'NegativeSamplingLoss',
    'evaluate_word_similarity',
    'evaluate_analogies',
    'compute_embedding_statistics',
    'get_nearest_neighbors',
    'compute_distortion_metrics',
    'compute_geometric_quality_metrics',
    'compute_capacity_metrics',
    'visualize_embeddings',
    'log_embeddings_to_wandb',
    'visualize_analogy',
    'create_manifold_geometry_plot',
    'train_embeddings',
]

__version__ = '0.1.0'

