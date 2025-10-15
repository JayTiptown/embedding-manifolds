import torch
from torch.optim import Optimizer
from manifolds import stiefel_project, sphere_project, stiefel_tangent_projection, sphere_tangent_projection


class ManifoldMuon(Optimizer):
    """
    Manifold Muon optimizer for Stiefel manifold.
    Solves dual problem: min_lambda max_||delta W|| <= 1 <G, delta W>
    Subject to W + delta W on Stiefel manifold.
    Reference: Blog Section 2
    """
    def __init__(self, params, lr=0.02, momentum=0.9, lambda_init=0.01, lambda_lr=0.01):
        defaults = dict(lr=lr, momentum=momentum, lambda_init=lambda_init, lambda_lr=lambda_lr)
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                state = self.state[p]
                
                if len(state) == 0:
                    state['momentum_buffer'] = torch.zeros_like(grad)
                    state['lambda'] = group['lambda_init']
                
                buf = state['momentum_buffer']
                lam = state['lambda']
                
                if p.dim() >= 2:
                    G_tan = stiefel_tangent_projection(p, grad)
                    buf.mul_(momentum).add_(G_tan, alpha=1 - momentum)
                    
                    spectral_norm = torch.linalg.matrix_norm(buf, ord=2)
                    normalized_update = buf / (spectral_norm + 1e-8)
                    
                    p.add_(normalized_update, alpha=-lr)
                    p.copy_(stiefel_project(p))
                    
                    constraint_violation = spectral_norm - 1.0
                    state['lambda'] = max(0, lam + group['lambda_lr'] * constraint_violation)
                else:
                    buf.mul_(momentum).add_(grad, alpha=1 - momentum)
                    p.add_(buf, alpha=-lr)
        
        return loss


class SphereOptimizer(Optimizer):
    """
    Optimizer for parameters on unit hypersphere.
    Reference: Blog Section 1
    """
    def __init__(self, params, lr=0.01, momentum=0.9):
        defaults = dict(lr=lr, momentum=momentum)
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                state = self.state[p]
                
                if len(state) == 0:
                    state['momentum_buffer'] = torch.zeros_like(grad)
                
                buf = state['momentum_buffer']
                
                g_tan = sphere_tangent_projection(p, grad)
                buf.mul_(momentum).add_(g_tan, alpha=1 - momentum)
                
                p.add_(buf, alpha=-lr)
                p.copy_(sphere_project(p))
        
        return loss


class StandardMuon(Optimizer):
    """
    Standard Muon optimizer with spectral normalization of updates.
    Baseline for comparison.
    """
    def __init__(self, params, lr=0.02, momentum=0.9):
        defaults = dict(lr=lr, momentum=momentum)
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                state = self.state[p]
                
                if len(state) == 0:
                    state['momentum_buffer'] = torch.zeros_like(grad)
                
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(grad, alpha=1 - momentum)
                
                if p.dim() >= 2:
                    spectral_norm = torch.linalg.matrix_norm(buf, ord=2)
                    normalized_update = buf / (spectral_norm + 1e-8)
                    p.add_(normalized_update, alpha=-lr)
                else:
                    p.add_(buf, alpha=-lr)
        
        return loss

