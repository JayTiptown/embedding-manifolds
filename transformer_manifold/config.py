"""
Experiment configurations for testing manifold constraints in transformers.

Architecture follows nanoGPT conventions for reproducibility.
Does constraining weight matrices to the Stiefel manifold improve training stability without hurting performance?
"""

from dataclasses import dataclass
from typing import Optional

@dataclass
class TransformerManifoldConfig:
    """
    Configuration for manifold constraint experiments.
    
    Default config matches nanoGPT's standard setup:
    - Architecture: GPT-2 style (384/6/6 for nano, 768/12/12 for small)
    - Hyperparams: Following nanoGPT conventions
    - Dataset: Character-level (256 vocab)
    """
    
    vocab_size: int = 256
    d_model: int = 384
    n_layers: int = 6
    n_heads: int = 6
    d_ff: int = 1536
    max_seq_len: int = 256
    dropout: float = 0.0
    bias: bool = True
    
    batch_size: int = 64
    gradient_accumulation_steps: int = 1
    num_epochs: int = 10
    learning_rate: float = 6e-4
    muon_lr: float = 0.02
    warmup_epochs: int = 1
    min_lr: float = 6e-5
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    
    projection_frequency: int = 10
    compile_model: bool = False
    track_condition_numbers: bool = True
    
    eval_interval: int = 500
    eval_iters: int = 200
    

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


def get_nano_config(**overrides):
    """
    nanoGPT-style config (~10M params).
    Default setup for reproducible experiments.
    """
    return TransformerManifoldConfig(
        d_model=384,
        n_layers=6,
        n_heads=6,
        d_ff=1536,
        max_seq_len=256,
        batch_size=64,
        gradient_accumulation_steps=4,
        **overrides
    )


def get_micro_config(**overrides):
    """
    Micro config for fast testing (~3M params).
    """
    return TransformerManifoldConfig(
        d_model=256,
        n_layers=4,
        n_heads=4,
        d_ff=1024,
        max_seq_len=256,
        batch_size=64,
        gradient_accumulation_steps=4,
        num_epochs=5,
        **overrides
    )


def get_small_config(**overrides):
    """
    Small GPT-2 style config (~50M params).
    """
    return TransformerManifoldConfig(
        d_model=768,
        n_layers=12,
        n_heads=12,
        d_ff=3072,
        max_seq_len=512,
        batch_size=32,
        gradient_accumulation_steps=8,
        **overrides
    )


def get_experiment_configs(size='nano'):
    """
    Generate all experiment configurations.
    
    Args:
        size: 'micro', 'nano', or 'small'
    
    Experiments (in order of complexity):
    1. Baseline: No constraints (Euclidean)
    2. FFN-only: Constrain feed-forward layers
    3. Attention-only: Constrain Q,K,V projections
    4. Full: Constrain both FFN and attention
    """
    if size == 'micro':
        config_fn = get_micro_config
    elif size == 'nano':
        config_fn = get_nano_config
    elif size == 'small':
        config_fn = get_small_config
    else:
        raise ValueError(f"Unknown size: {size}. Choose 'micro', 'nano', or 'small'")
    
    configs = [
        config_fn(
            constraint_ffn=False,
            constraint_attention=False,
            manifold_type='euclidean'
        ),
        
        config_fn(
            constraint_ffn=True,
            constraint_attention=False,
            manifold_type='stiefel'
        ),
        
        config_fn(
            constraint_ffn=False,
            constraint_attention=True,
            manifold_type='stiefel'
        ),
        
        config_fn(
            constraint_ffn=True,
            constraint_attention=True,
            manifold_type='stiefel'
        ),
    ]
    
    return configs