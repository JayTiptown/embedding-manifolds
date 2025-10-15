import numpy as np
import torch
import wandb
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.graph_objects as go


def reduce_dimensions(embeddings, method='pca', n_components=3):
    """
    Reduce embeddings to lower dimensions for visualization.
    
    Args:
        embeddings: (n_words, embedding_dim) numpy array
        method: 'pca' or 'tsne'
        n_components: 2 or 3 for visualization
    
    Returns:
        reduced: (n_words, n_components) numpy array
    """
    if method == 'pca':
        reducer = PCA(n_components=n_components)
    elif method == 'tsne':
        reducer = TSNE(n_components=n_components, random_state=42, perplexity=min(30, len(embeddings)-1))
    else:
        raise ValueError(f"Unknown method: {method}")
    
    reduced = reducer.fit_transform(embeddings)
    return reduced


def create_2d_scatter(embeddings, words, title="Word Embeddings 2D"):
    """
    Create 2D scatter plot of embeddings.
    
    Args:
        embeddings: (n_words, 2) numpy array
        words: list of word strings
        title: plot title
    
    Returns:
        plotly figure
    """
    fig = go.Figure(data=[
        go.Scatter(
            x=embeddings[:, 0],
            y=embeddings[:, 1],
            mode='markers+text',
            text=words,
            textposition='top center',
            marker=dict(
                size=8,
                color=np.arange(len(words)),
                colorscale='Viridis',
                showscale=True
            ),
            hovertext=words,
            hoverinfo='text'
        )
    ])
    
    fig.update_layout(
        title=title,
        xaxis_title='Dimension 1',
        yaxis_title='Dimension 2',
        hovermode='closest',
        width=1000,
        height=800
    )
    
    return fig


def create_3d_scatter(embeddings, words, title="Word Embeddings 3D"):
    """
    Create 3D scatter plot of embeddings.
    
    Args:
        embeddings: (n_words, 3) numpy array
        words: list of word strings
        title: plot title
    
    Returns:
        plotly figure
    """
    fig = go.Figure(data=[
        go.Scatter3d(
            x=embeddings[:, 0],
            y=embeddings[:, 1],
            z=embeddings[:, 2],
            mode='markers+text',
            text=words,
            textposition='top center',
            marker=dict(
                size=6,
                color=np.arange(len(words)),
                colorscale='Viridis',
                showscale=True,
                opacity=0.8
            ),
            hovertext=words,
            hoverinfo='text'
        )
    ])
    
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='Dimension 1',
            yaxis_title='Dimension 2',
            zaxis_title='Dimension 3'
        ),
        hovermode='closest',
        width=1200,
        height=900
    )
    
    return fig


def visualize_embeddings(model, vocab, sample_words=None, n_words=50, reduction_method='pca'):
    """
    Visualize word embeddings in 2D and 3D.
    
    Args:
        model: WordEmbeddings model
        vocab: Vocabulary object
        sample_words: specific words to visualize (None for random)
        n_words: number of words to visualize if sample_words is None
        reduction_method: 'pca' or 'tsne'
    
    Returns:
        dict with 2D and 3D figures
    """
    embeddings = model.get_embeddings()
    
    if sample_words is not None:
        indices = [vocab.get_idx(w) for w in sample_words if vocab.get_idx(w) != 0]
        words = [w for w in sample_words if vocab.get_idx(w) != 0]
    else:
        valid_indices = list(range(1, min(len(vocab), embeddings.shape[0])))
        np.random.shuffle(valid_indices)
        indices = valid_indices[:n_words]
        words = [vocab.idx2word[i] for i in indices]
    
    selected_embeddings = embeddings[indices]
    
    embeddings_2d = reduce_dimensions(selected_embeddings, method=reduction_method, n_components=2)
    embeddings_3d = reduce_dimensions(selected_embeddings, method=reduction_method, n_components=3)
    
    fig_2d = create_2d_scatter(embeddings_2d, words, 
                               title=f"Word Embeddings 2D ({model.manifold.upper()}, {reduction_method.upper()})")
    fig_3d = create_3d_scatter(embeddings_3d, words,
                               title=f"Word Embeddings 3D ({model.manifold.upper()}, {reduction_method.upper()})")
    
    return {
        '2d': fig_2d,
        '3d': fig_3d,
        'embeddings_2d': embeddings_2d,
        'embeddings_3d': embeddings_3d,
        'words': words
    }


def log_embeddings_to_wandb(model, vocab, step, sample_words=None, n_words=50):
    """
    Log embedding visualizations to wandb.
    
    Args:
        model: WordEmbeddings model
        vocab: Vocabulary object
        step: training step/epoch
        sample_words: specific words to visualize
        n_words: number of words to visualize
    """
    vis_pca = visualize_embeddings(model, vocab, sample_words, n_words, reduction_method='pca')
    vis_tsne = visualize_embeddings(model, vocab, sample_words, n_words, reduction_method='tsne')
    
    wandb.log({
        f'embeddings/pca_2d': wandb.Plotly(vis_pca['2d']),
        f'embeddings/pca_3d': wandb.Plotly(vis_pca['3d']),
        f'embeddings/tsne_2d': wandb.Plotly(vis_tsne['2d']),
        f'embeddings/tsne_3d': wandb.Plotly(vis_tsne['3d']),
        'step': step
    })


def visualize_analogy(model, vocab, a, b, c, predicted_words, title="Analogy Visualization"):
    """
    Visualize analogy relationships in embedding space.
    
    Args:
        model: WordEmbeddings model
        vocab: Vocabulary object
        a, b, c: words in analogy "a is to b as c is to ?"
        predicted_words: list of predicted words for the analogy
        title: plot title
    
    Returns:
        plotly figure
    """
    words = [a, b, c] + predicted_words
    indices = [vocab.get_idx(w) for w in words]
    embeddings = model.get_embeddings()[indices]
    
    embeddings_3d = reduce_dimensions(embeddings, method='pca', n_components=3)
    
    colors = ['red', 'red', 'blue'] + ['green'] * len(predicted_words)
    sizes = [12, 12, 12] + [8] * len(predicted_words)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter3d(
        x=embeddings_3d[:, 0],
        y=embeddings_3d[:, 1],
        z=embeddings_3d[:, 2],
        mode='markers+text',
        text=words,
        textposition='top center',
        marker=dict(
            size=sizes,
            color=colors,
            opacity=0.8
        ),
        hovertext=words,
        hoverinfo='text'
    ))
    
    fig.add_trace(go.Scatter3d(
        x=[embeddings_3d[0, 0], embeddings_3d[1, 0]],
        y=[embeddings_3d[0, 1], embeddings_3d[1, 1]],
        z=[embeddings_3d[0, 2], embeddings_3d[1, 2]],
        mode='lines',
        line=dict(color='red', width=4),
        name='a → b'
    ))
    
    fig.add_trace(go.Scatter3d(
        x=[embeddings_3d[2, 0], embeddings_3d[3, 0]],
        y=[embeddings_3d[2, 1], embeddings_3d[3, 1]],
        z=[embeddings_3d[2, 2], embeddings_3d[3, 2]],
        mode='lines',
        line=dict(color='blue', width=4),
        name='c → predicted'
    ))
    
    fig.update_layout(
        title=f"{title}: {a}:{b} :: {c}:?",
        scene=dict(
            xaxis_title='Dimension 1',
            yaxis_title='Dimension 2',
            zaxis_title='Dimension 3'
        ),
        width=1200,
        height=900
    )
    
    return fig


def create_manifold_geometry_plot(model, vocab, n_points=100):
    """
    Create visualization showing the geometry of the manifold.
    For sphere: show points on unit sphere
    For Stiefel: show orthogonality properties
    
    Args:
        model: WordEmbeddings model
        vocab: Vocabulary object
        n_points: number of points to sample
    
    Returns:
        plotly figure
    """
    embeddings = model.get_embeddings()
    
    valid_indices = list(range(1, min(len(vocab), embeddings.shape[0])))
    np.random.shuffle(valid_indices)
    indices = valid_indices[:n_points]
    
    sampled_embeddings = embeddings[indices]
    embeddings_3d = reduce_dimensions(sampled_embeddings, method='pca', n_components=3)
    
    norms = np.linalg.norm(sampled_embeddings, axis=1)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter3d(
        x=embeddings_3d[:, 0],
        y=embeddings_3d[:, 1],
        z=embeddings_3d[:, 2],
        mode='markers',
        marker=dict(
            size=5,
            color=norms,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title='Norm'),
            opacity=0.7
        ),
        hovertext=[f"norm: {n:.3f}" for n in norms],
        hoverinfo='text'
    ))
    
    if model.manifold == 'sphere':
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 50)
        x_sphere = np.outer(np.cos(u), np.sin(v))
        y_sphere = np.outer(np.sin(u), np.sin(v))
        z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
        
        sphere_points = np.stack([x_sphere.flatten(), y_sphere.flatten(), z_sphere.flatten()], axis=1)
        sphere_3d = reduce_dimensions(sphere_points, method='pca', n_components=3)
        sphere_3d = sphere_3d.reshape(50, 50, 3)
        
        fig.add_trace(go.Surface(
            x=sphere_3d[:, :, 0],
            y=sphere_3d[:, :, 1],
            z=sphere_3d[:, :, 2],
            opacity=0.3,
            colorscale='Blues',
            showscale=False,
            name='Unit Sphere'
        ))
    
    fig.update_layout(
        title=f"Manifold Geometry: {model.manifold.upper()}",
        scene=dict(
            xaxis_title='PC 1',
            yaxis_title='PC 2',
            zaxis_title='PC 3',
            aspectmode='cube'
        ),
        width=1200,
        height=900
    )
    
    return fig

