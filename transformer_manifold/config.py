"""
Experiment configurations for testing manifold constraints in transformers.

Does constraining weight matrices to the Stiefel manifold improve training stability without hurting performance?
"""

from dataclasses import dataclass
from typing import Optional

@dataclass
class TransformerManifoldConfig:
    """Configuration for manifold constraint experiments."""
    
    vocab_size: int = 256
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 4
    d_ff: int = 1024
    max_seq_len: int = 256
    dropout: float = 0.1
    
    batch_size: int = 32
    num_epochs: int = 10
    learning_rate: float = 3e-4
    muon_lr: float = 0.02
    
    eval_interval: int = 100
    eval_iters: int = 50
    

    constraint_ffn: bool = False 
    constraint_attention: bool = False 
    manifold_type: str = 'stiefel' 
    
    wandb_project: str = "transformer-manifolds"
    wandb_entity: Optional[str] = None
    
    device: str = 'cuda' 
    seed: int = 42
    
    def get_name(self):
        """Generate experiment name from config."""
        parts = []
        
        if not self.constraint_ffn and not self.constraint_attention:
            return "baseline_euclidean"
        
        if self.constraint_ffn:
            parts.append("ffn")
        if self.constraint_attention:
            parts.append("attn")
        
        location = "_".join(parts)
        return f"{location}_{self.manifold_type}"
    
    def get_optimizer_type(self):
        """Determine which optimizer to use based on constraints."""
        if self.manifold_type == 'euclidean':
            return 'adam'
        elif self.manifold_type == 'stiefel':
            return 'manifold_muon'
        elif self.manifold_type == 'sphere':
            return 'sphere_optimizer'
        else:
            raise ValueError(f"Unknown manifold type: {self.manifold_type}")


def get_experiment_configs():
    """
    Generate all experiment configurations.
    
    Experiments (in order of complexity):
    1. Baseline: No constraints (Euclidean)
    2. FFN-only: Constrain feed-forward layers
    3. Attention-only: Constrain Q,K,V projections
    4. Full: Constrain both FFN and attention
    """
    base_config = TransformerManifoldConfig()
    
    configs = [
        TransformerManifoldConfig(
            constraint_ffn=False,
            constraint_attention=False,
            manifold_type='euclidean'
        ),
        
        TransformerManifoldConfig(
            constraint_ffn=True,
            constraint_attention=False,
            manifold_type='stiefel'
        ),
        
        TransformerManifoldConfig(
            constraint_ffn=False,
            constraint_attention=True,
            manifold_type='stiefel'
        ),
        
        TransformerManifoldConfig(
            constraint_ffn=True,
            constraint_attention=True,
            manifold_type='stiefel'
        ),
    ]
    
    return configs