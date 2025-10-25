"""
Minimal transformer with optional manifold constraints.

Key design decision: Embeddings stay Euclidean (your experiments showed this is best),
but weight matrices can be constrained to manifolds.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from manifolds import stiefel_project, sphere_project


class ManifoldLinear(nn.Module):
    """
    Linear layer with optional manifold constraint.
    
    When constrained=True, weight matrix is projected to manifold after each update.
    Input and output remain Euclidean.
    """
    def __init__(self, in_features, out_features, constrained=False, manifold='stiefel', bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.constrained = constrained
        self.manifold = manifold
        
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.weight)
        
        if self.constrained:
            with torch.no_grad():
                if self.manifold == 'stiefel' and self.out_features <= self.in_features:
                    self.weight.copy_(stiefel_project(self.weight))
                elif self.manifold == 'sphere':
                    self.weight.copy_(F.normalize(self.weight, p=2, dim=1))
    
    def forward(self, x):
        return F.linear(x, self.weight, self.bias)
    
    def project_to_manifold(self):
        """Project weight matrix back to manifold."""
        if not self.constrained:
            return
        
        with torch.no_grad():
            if self.manifold == 'stiefel' and self.out_features <= self.in_features:
                self.weight.copy_(stiefel_project(self.weight))
            elif self.manifold == 'sphere':
                self.weight.copy_(F.normalize(self.weight, p=2, dim=1))
    
    def get_condition_number(self):
        """Compute condition number of weight matrix."""
        with torch.no_grad():
            if self.weight.dim() < 2:
                return None
            S = torch.linalg.svdvals(self.weight)
            if S.numel() == 0:
                return None
            return (S[0] / (S[-1] + 1e-8)).item()


class MultiHeadAttention(nn.Module):
    """Multi-head attention with optional manifold constraints on Q,K,V."""
    def __init__(self, d_model, n_heads, dropout=0.1, constrained=False, manifold='stiefel'):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.q_proj = ManifoldLinear(d_model, d_model, constrained, manifold)
        self.k_proj = ManifoldLinear(d_model, d_model, constrained, manifold)
        self.v_proj = ManifoldLinear(d_model, d_model, constrained, manifold)
        self.out_proj = ManifoldLinear(d_model, d_model, constrained, manifold)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.shape
        
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        return self.out_proj(out)
    
    def project_to_manifold(self):
        """Project all weight matrices to manifold."""
        self.q_proj.project_to_manifold()
        self.k_proj.project_to_manifold()
        self.v_proj.project_to_manifold()
        self.out_proj.project_to_manifold()


class FeedForward(nn.Module):
    """Feed-forward network with optional manifold constraints."""
    def __init__(self, d_model, d_ff, dropout=0.1, constrained=False, manifold='stiefel'):
        super().__init__()
        self.fc1 = ManifoldLinear(d_model, d_ff, constrained, manifold)
        self.fc2 = ManifoldLinear(d_ff, d_model, constrained, manifold)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.fc2(self.dropout(F.gelu(self.fc1(x))))
    
    def project_to_manifold(self):
        """Project all weight matrices to manifold."""
        self.fc1.project_to_manifold()
        self.fc2.project_to_manifold()


class TransformerBlock(nn.Module):
    """Single transformer block."""
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, 
                 constraint_ffn=False, constraint_attention=False, manifold='stiefel'):
        super().__init__()
        
        self.attn = MultiHeadAttention(d_model, n_heads, dropout, constraint_attention, manifold)
        self.ff = FeedForward(d_model, d_ff, dropout, constraint_ffn, manifold)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        x = x + self.dropout(self.attn(self.norm1(x), mask))
        
        x = x + self.dropout(self.ff(self.norm2(x)))
        
        return x
    
    def project_to_manifold(self):
        """Project all constrained weights to manifold."""
        self.attn.project_to_manifold()
        self.ff.project_to_manifold()


class ManifoldTransformer(nn.Module):
    """
    Transformer with optional manifold constraints.
    
    Design: Embeddings stay Euclidean (your experiments showed this works best),
    but weight matrices can be constrained to Stiefel/Sphere manifolds.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Embeddings: ALWAYS Euclidean
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, config.max_seq_len, config.d_model))
        
        # Transformer blocks with optional constraints
        self.blocks = nn.ModuleList([
            TransformerBlock(
                config.d_model,
                config.n_heads,
                config.d_ff,
                config.dropout,
                constraint_ffn=config.constraint_ffn,
                constraint_attention=config.constraint_attention,
                manifold=config.manifold_type
            )
            for _ in range(config.n_layers)
        ])
        
        self.norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size)
        self.dropout = nn.Dropout(config.dropout)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.embedding.weight, mean=0, std=0.02)
        nn.init.normal_(self.pos_embedding, mean=0, std=0.02)
    
    def forward(self, x, targets=None):
        batch_size, seq_len = x.shape
        
        tok_emb = self.embedding(x)
        pos_emb = self.pos_embedding[:, :seq_len, :]
        x = self.dropout(tok_emb + pos_emb)

        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).view(1, 1, seq_len, seq_len)
        
        for block in self.blocks:
            x = block(x, mask)
        
        x = self.norm(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        return logits, loss
    
    def project_to_manifold(self):
        """Project all constrained weights to their manifolds."""
        for block in self.blocks:
            block.project_to_manifold()
    
    def get_all_condition_numbers(self):
        """
        Get condition numbers for all weight matrices.
        
        Returns:
            dict: {layer_name: condition_number}
        """
        cond_numbers = {}
        
        for name, module in self.named_modules():
            if isinstance(module, ManifoldLinear):
                cond = module.get_condition_number()
                if cond is not None:
                    cond_numbers[name] = cond
        
        return cond_numbers
    
    @torch.no_grad()
    def get_condition_number_stats(self):
        """Get statistics about condition numbers."""
        cond_numbers = self.get_all_condition_numbers()
        
        if not cond_numbers:
            return {'mean': 0, 'max': 0, 'min': 0, 'std': 0, 'by_layer': {}}
        
        values = list(cond_numbers.values())
        mean_val = sum(values) / len(values)
        return {
            'mean': mean_val,
            'max': max(values),
            'min': min(values),
            'std': (sum((x - mean_val)**2 for x in values) / len(values)) ** 0.5,
            'count': len(values),
            'by_layer': cond_numbers
        }