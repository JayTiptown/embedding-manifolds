import torch
import torch.nn as nn
from manifolds import sphere_project, stiefel_project


class WordEmbeddings(nn.Module):
    """
    Word embedding model with different manifold constraints.
    
    Supports:
    - 'euclidean': Standard unconstrained embeddings
    - 'sphere': Each embedding vector on unit hypersphere
    - 'stiefel': Embedding matrix on Stiefel manifold (orthogonal rows)
    """
    def __init__(self, vocab_size, embedding_dim, manifold='euclidean'):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.manifold = manifold
        
        self.embeddings = nn.Parameter(torch.randn(vocab_size, embedding_dim) * 0.1)
        self.context_embeddings = nn.Parameter(torch.randn(vocab_size, embedding_dim) * 0.1)
        
        self._init_embeddings()
    
    def _init_embeddings(self):
        """Initialize embeddings on the manifold."""
        with torch.no_grad():
            if self.manifold == 'sphere':
                self.embeddings.copy_(sphere_project(self.embeddings))
                self.context_embeddings.copy_(sphere_project(self.context_embeddings))
            elif self.manifold == 'stiefel':
                self.embeddings.copy_(stiefel_project(self.embeddings))
                self.context_embeddings.copy_(stiefel_project(self.context_embeddings))
    
    def project_to_manifold(self):
        """Project embeddings back to manifold after gradient update."""
        if self.manifold == 'euclidean':
            return
        
        with torch.no_grad():
            if self.manifold == 'sphere':
                self.embeddings.copy_(sphere_project(self.embeddings))
                self.context_embeddings.copy_(sphere_project(self.context_embeddings))
            elif self.manifold == 'stiefel':
                self.embeddings.copy_(stiefel_project(self.embeddings))
                self.context_embeddings.copy_(stiefel_project(self.context_embeddings))
    
    def forward(self, center_ids, context_ids):
        """
        Forward pass for skip-gram.
        
        Args:
            center_ids: (batch_size,) center word indices
            context_ids: (batch_size,) context word indices
        
        Returns:
            scores: (batch_size,) similarity scores
        """
        center_vecs = self.embeddings[center_ids]
        context_vecs = self.context_embeddings[context_ids]
        
        scores = (center_vecs * context_vecs).sum(dim=1)
        
        return scores
    
    def get_embeddings(self):
        """Return the word embeddings (not context embeddings)."""
        return self.embeddings.data.cpu().numpy()
    
    def similarity(self, idx1, idx2):
        """Compute similarity between two word embeddings."""
        v1 = self.embeddings[idx1]
        v2 = self.embeddings[idx2]
        
        if self.manifold == 'sphere':
            return torch.dot(v1, v2)
        else:
            return torch.dot(v1, v2) / (torch.norm(v1) * torch.norm(v2) + 1e-8)
    
    def analogy(self, a_idx, b_idx, c_idx, k=5):
        """
        Solve analogy: a is to b as c is to ?.
        
        Returns top k candidates by computing: d = b - a + c
        and finding nearest neighbors to d.
        """
        a = self.embeddings[a_idx]
        b = self.embeddings[b_idx]
        c = self.embeddings[c_idx]
        
        d = b - a + c
        
        if self.manifold == 'sphere':
            d = d / (torch.norm(d) + 1e-8)
        
        scores = torch.matmul(self.embeddings, d)
        
        if self.manifold != 'sphere':
            norms = torch.norm(self.embeddings, dim=1)
            scores = scores / (norms * torch.norm(d) + 1e-8)
        
        scores[a_idx] = -float('inf')
        scores[b_idx] = -float('inf')
        scores[c_idx] = -float('inf')
        
        top_k = torch.topk(scores, k)
        return top_k.indices.cpu().numpy(), top_k.values.cpu().numpy()


class NegativeSamplingLoss(nn.Module):
    """Negative sampling loss for skip-gram."""
    def __init__(self, num_negatives=5):
        super().__init__()
        self.num_negatives = num_negatives
    
    def forward(self, model, center_ids, context_ids, vocab_size):
        """
        Compute negative sampling loss.
        
        Args:
            model: WordEmbeddings model
            center_ids: (batch_size,) center word indices
            context_ids: (batch_size,) context word indices
            vocab_size: size of vocabulary
        
        Returns:
            loss: negative sampling loss
        """
        batch_size = center_ids.size(0)
        
        positive_scores = model(center_ids, context_ids)
        positive_loss = -torch.log(torch.sigmoid(positive_scores) + 1e-8).mean()
        
        negative_loss = 0
        for _ in range(self.num_negatives):
            negative_ids = torch.randint(0, vocab_size, (batch_size,), device=center_ids.device)
            negative_scores = model(center_ids, negative_ids)
            negative_loss += -torch.log(torch.sigmoid(-negative_scores) + 1e-8).mean()
        
        negative_loss /= self.num_negatives
        
        return positive_loss + negative_loss

