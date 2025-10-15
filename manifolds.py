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
