import torch
import torch.nn.functional as F


def matrix_sign(W, max_iter=20, tol=1e-6):
    """
    Compute matrix sign using Newton-Schulz iteration.
    Z_{k+1} = 0.5(3Z_k - Z_k^3) where Z_0 = W/||W||_spectral
    """
    with torch.no_grad():
        spectral_norm = torch.linalg.matrix_norm(W, ord=2)
        Z = W / (spectral_norm + 1e-8)
        
        for _ in range(max_iter):
            Z_prev = Z
            Z = 0.5 * (3 * Z - Z @ Z @ Z)
            if torch.norm(Z - Z_prev) < tol:
                break
        
        return Z


def stiefel_project(W):
    """
    Project matrix W to Stiefel manifold using matrix sign.
    Returns W @ sign(W^T W)^{-1/2}
    """
    with torch.no_grad():
        WtW = W.T @ W
        sign_WtW = matrix_sign(WtW)
        
        U, S, Vh = torch.linalg.svd(sign_WtW, full_matrices=False)
        inv_sqrt = U @ torch.diag(1.0 / torch.sqrt(S + 1e-8)) @ Vh
        
        return W @ inv_sqrt


def sphere_project(v):
    """Project vector v to unit hypersphere."""
    with torch.no_grad():
        return F.normalize(v, p=2, dim=-1)


def stiefel_tangent_projection(W, G):
    """
    Project gradient G to tangent space of Stiefel manifold at W.
    T_W(Stiefel) = {G : W^T G + G^T W = 0}
    Projection: G - W(W^T G + G^T W)/2
    """
    sym = W.T @ G
    return G - W @ (sym + sym.T) / 2


def sphere_tangent_projection(v, g):
    """
    Project gradient g to tangent space of hypersphere at v.
    T_v(Sphere) = {g : v^T g = 0}
    Projection: g - v(v^T g)
    """
    return g - v * (v * g).sum(dim=-1, keepdim=True)


def sphere_geodesic_distance(v1, v2):
    """
    Compute geodesic distance on unit hypersphere.
    Distance is the angle between vectors: d(v1, v2) = arccos(v1 · v2)
    
    Args:
        v1, v2: unit norm vectors on hypersphere
    
    Returns:
        geodesic distance (angle in radians)
    """
    dot_product = (v1 * v2).sum(dim=-1)
    dot_product = torch.clamp(dot_product, -1.0, 1.0)
    return torch.acos(dot_product)


def sphere_exponential_map(v, tangent_vec):
    """
    Exponential map on unit hypersphere.
    Maps tangent vector at v to point on sphere via geodesic.
    
    exp_v(ξ) = cos(||ξ||)v + sin(||ξ||)(ξ/||ξ||)
    
    Args:
        v: base point on sphere (unit norm)
        tangent_vec: tangent vector at v
    
    Returns:
        point on sphere reached by following geodesic
    """
    with torch.no_grad():
        norm = torch.norm(tangent_vec, dim=-1, keepdim=True)
        
        mask = norm.squeeze(-1) > 1e-8
        result = v.clone()
        
        if mask.any():
            cos_norm = torch.cos(norm)
            sin_norm = torch.sin(norm)
            direction = tangent_vec / (norm + 1e-8)
            
            result_new = cos_norm * v + sin_norm * direction
            result[mask] = result_new[mask]
        
        return result
