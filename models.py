import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from manifolds import stiefel_project, sphere_project


class ConstrainedLinear(nn.Module):
    """Linear layer with optional Stiefel manifold constraint."""
    def __init__(self, in_features, out_features, constrained=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.constrained = constrained
        
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.weight)
        if self.constrained and self.out_features <= self.in_features:
            with torch.no_grad():
                self.weight.copy_(stiefel_project(self.weight))
    
    def forward(self, x):
        return F.linear(x, self.weight, self.bias)


class ConstrainedEmbedding(nn.Module):
    """Embedding layer with optional hypersphere constraint per vector."""
    def __init__(self, num_embeddings, embedding_dim, constrained=False):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.constrained = constrained
        
        self.weight = nn.Parameter(torch.randn(num_embeddings, embedding_dim))
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.weight, mean=0, std=0.02)
        if self.constrained:
            with torch.no_grad():
                self.weight.copy_(sphere_project(self.weight))
    
    def forward(self, x):
        return F.embedding(x, self.weight)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1, constrained=False):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.q_proj = ConstrainedLinear(d_model, d_model, constrained)
        self.k_proj = ConstrainedLinear(d_model, d_model, constrained)
        self.v_proj = ConstrainedLinear(d_model, d_model, constrained)
        self.out_proj = ConstrainedLinear(d_model, d_model, constrained)
        
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


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1, constrained=False):
        super().__init__()
        self.fc1 = ConstrainedLinear(d_model, d_ff, constrained)
        self.fc2 = ConstrainedLinear(d_ff, d_model, constrained)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.fc2(self.dropout(F.gelu(self.fc1(x))))


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, constrained=False):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads, dropout, constrained)
        self.ff = FeedForward(d_model, d_ff, dropout, constrained)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        x = x + self.dropout(self.attn(self.norm1(x), mask))
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x


class SmallTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, 
                 max_seq_len, dropout=0.1, constrained=False):
        super().__init__()
        self.d_model = d_model
        self.constrained = constrained
        
        self.embedding = ConstrainedEmbedding(vocab_size, d_model, constrained)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, d_model))
        
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout, constrained)
            for _ in range(n_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = ConstrainedLinear(d_model, vocab_size, constrained)
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()
    
    def _init_weights(self):
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
    
    def get_condition_numbers(self):
        """Compute condition numbers of weight matrices."""
        cond_numbers = []
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() >= 2 and param.requires_grad:
                with torch.no_grad():
                    try:
                        S = torch.linalg.svdvals(param)
                        cond = (S.max() / (S.min() + 1e-8)).item()
                        cond_numbers.append((name, cond))
                    except:
                        pass
        return cond_numbers

