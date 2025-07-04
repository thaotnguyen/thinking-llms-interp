"""
Clustering utilities for neural activation analysis.
This module contains common functions for clustering analysis, model saving/loading,
and evaluation metrics.
"""

import os
import sys
import numpy as np
import pickle
import time
from sklearn.metrics import silhouette_score


def print_and_flush(message):
    """Prints a message and flushes stdout."""
    print(message)
    sys.stdout.flush()


def load_clustering_model(model_id, layer, n_clusters, method):
    """
    Load a saved clustering model from pickle file.
    
    Parameters:
    -----------
    model_id : str
        Model identifier (e.g., "deepseek-r1-distill-qwen-1.5b")
    layer : int
        Layer number
    n_clusters : int
        Number of clusters
    method : str
        Clustering method name
        
    Returns:
    --------
    dict
        Dictionary containing the loaded clustering data
    """
    clustering_path = f'results/vars/clustering_models/{method}_{model_id}_layer{layer}_clusters{n_clusters}.pkl'
    if not os.path.exists(clustering_path):
        raise FileNotFoundError(f"Clustering model not found at {clustering_path}")
        
    with open(clustering_path, 'rb') as f:
        clustering_data = pickle.load(f)
    
    print_and_flush(f"Loaded {method} clustering model from {clustering_path}")
    return clustering_data


def predict_clusters(new_data, clustering_data):
    """
    Predict cluster labels for new data using a saved clustering model.
    
    Parameters:
    -----------
    new_data : numpy.ndarray
        New data to cluster (should be normalized)
    clustering_data : dict
        Dictionary containing the loaded clustering data
        
    Returns:
    --------
    numpy.ndarray
        Predicted cluster labels
    """
    method = clustering_data['method']
    model = clustering_data['model']
    
    if method == 'agglomerative':
        # Agglomerative clustering doesn't have a predict method, so we need to compute distances
        cluster_centers = clustering_data['cluster_centers']
        # Compute distances to all cluster centers
        distances = np.zeros((new_data.shape[0], len(cluster_centers)))
        for i, center in enumerate(cluster_centers):
            distances[:, i] = np.linalg.norm(new_data - center, axis=1)
        # Assign to closest cluster
        cluster_labels = np.argmin(distances, axis=1)
        
    elif method in ['spherical_kmeans', 'pca_kmeans']:
        if method == 'spherical_kmeans':
            # Normalize data for spherical kmeans
            new_data_norm = new_data / np.linalg.norm(new_data, axis=1, keepdims=True)
            cluster_labels = model.predict(new_data_norm)
        else:
            # For PCA methods, we need to apply PCA first
            pca = clustering_data['pca']
            new_data_reduced = pca.transform(new_data)
            cluster_labels = model.predict(new_data_reduced)
            
    elif method in ['gmm', 'pca_gmm']:
        if method == 'gmm':
            cluster_labels = model.predict(new_data)
        else:
            # For PCA methods, we need to apply PCA first
            pca = clustering_data['pca']
            new_data_reduced = pca.transform(new_data)
            cluster_labels = model.predict(new_data_reduced)
            
    elif method == 'pca_agglomerative':
        # Agglomerative clustering doesn't have a predict method
        pca = clustering_data['pca']
        new_data_reduced = pca.transform(new_data)
        cluster_centers = clustering_data['cluster_centers']
        # Compute distances to all cluster centers in reduced space
        distances = np.zeros((new_data_reduced.shape[0], len(cluster_centers)))
        for i, center in enumerate(cluster_centers):
            distances[:, i] = np.linalg.norm(new_data_reduced - center, axis=1)
        # Assign to closest cluster
        cluster_labels = np.argmin(distances, axis=1)
        
    elif method == 'sae_topk':
        # For SAE models, we need to load the SAE and use the encoder
        import torch
        from utils import utils
        
        # Load the SAE model
        sae_path = clustering_data.get('sae_path')
        if sae_path is None:
            # Try to construct the path
            model_id = clustering_data.get('model_id', 'unknown')
            layer = clustering_data.get('layer', 0)
            n_clusters = clustering_data.get('n_clusters', 0)
            sae_path = f'results/vars/clustering_models/sae_{model_id}_layer{layer}_clusters{n_clusters}.pt'
        
        if not os.path.exists(sae_path):
            raise FileNotFoundError(f"SAE model not found at {sae_path}")
        
        # Load SAE checkpoint
        checkpoint = torch.load(sae_path, map_location='cpu')
        
        # Create SAE model
        sae = utils.SAE(checkpoint['input_dim'], checkpoint['num_latents'], k=checkpoint.get('topk', 3))
        
        # Load weights
        sae.encoder.weight.data = checkpoint['encoder_weight']
        sae.encoder.bias.data = checkpoint['encoder_bias']
        sae.W_dec.data = checkpoint['decoder_weight']
        sae.b_dec.data = checkpoint['b_dec']
        
        # Convert input data to torch tensor
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X = torch.from_numpy(new_data).float().to(device)
        
        # Use the encoder to get cluster assignments
        sae.eval()
        with torch.no_grad():
            # Get the full activations without topk restriction for cluster assignment
            activations = sae.encoder(X - sae.b_dec)
            # Get the index of the maximum activation for each example
            cluster_labels = activations.argmax(dim=1).cpu().numpy()
        
        # Clean up
        del sae, X
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
    else:
        raise ValueError(f"Unknown clustering method: {method}")
    
    return cluster_labels


def compute_centroid_orthogonality(cluster_centers):
    """
    Compute the orthogonality of cluster centroids using 1 - cosine similarity.
    Uses pairwise_distances from sklearn to explicitly compute all pairwise similarities.
    
    Parameters:
    -----------
    cluster_centers : numpy.ndarray
        Cluster center vectors
        
    Returns:
    --------
    float
        Average orthogonality (1 - cosine similarity) between centroids
    """
    # First compute cosine similarity (not distance)
    norm_cluster_centers = cluster_centers / np.linalg.norm(cluster_centers, axis=1, keepdims=True)
    # Use dot product for cosine similarity
    cosine_sim = np.dot(norm_cluster_centers, norm_cluster_centers.T)
    # Take absolute value to treat opposite directions as similar
    abs_cosine_sim = np.abs(cosine_sim)
    # Calculate orthogonality as 1 - absolute similarity
    orthogonality = 1 - abs_cosine_sim
    
    # Get the indices of the upper triangular part (excluding diagonal)
    # This ensures we only count each pair once and exclude self-similarities
    indices = np.triu_indices(orthogonality.shape[0], k=1)
    
    # Extract the upper triangular values
    upper_tri_values = orthogonality[indices]
    
    # Calculate average orthogonality
    avg_orthogonality = np.mean(upper_tri_values) if len(upper_tri_values) > 0 else 0.0
    
    return avg_orthogonality


def save_clustering_model(clustering_data, model_id, layer, n_clusters, method):
    """
    Save a clustering model to pickle file.
    
    Parameters:
    -----------
    clustering_data : dict
        Dictionary containing the clustering data to save
    model_id : str
        Model identifier
    layer : int
        Layer number
    n_clusters : int
        Number of clusters
    method : str
        Clustering method name
    """
    # Create directory for saving clustering models
    os.makedirs('results/vars/clustering_models', exist_ok=True)
    
    # Save the clustering model
    clustering_save_path = f'results/vars/clustering_models/{method}_{model_id}_layer{layer}_clusters{n_clusters}.pkl'
    
    with open(clustering_save_path, 'wb') as f:
        pickle.dump(clustering_data, f)
    print_and_flush(f"Saved {method} clustering model to {clustering_save_path}")


def compute_silhouette_score(data, cluster_labels, sample_size=50000, random_state=42):
    """
    Compute silhouette score with timing information.
    
    Parameters:
    -----------
    data : numpy.ndarray
        Data used for clustering
    cluster_labels : numpy.ndarray
        Cluster labels
    sample_size : int
        Number of samples to use for silhouette score calculation
    random_state : int
        Random state for reproducibility
        
    Returns:
    --------
    float
        Silhouette score
    """
    silhouette_start_time = time.time()
    silhouette = silhouette_score(data, cluster_labels, sample_size=sample_size, random_state=random_state)
    print_and_flush(f"    Silhouette score calculation completed in {time.time() - silhouette_start_time:.2f} seconds")
    return silhouette


def generate_test_data(input_dim, n_samples=10000):
    """
    Generate test data for evaluation.
    
    Parameters:
    -----------
    input_dim : int
        Input dimension
    n_samples : int
        Number of test samples
        
    Returns:
    --------
    numpy.ndarray
        Normalized test data
    """
    # Generate random test data
    test_data = np.random.randn(n_samples, input_dim)
    # Normalize the test data
    test_data = test_data / np.linalg.norm(test_data, axis=1, keepdims=True)
    return test_data


def convert_numpy_types(obj):
    """
    Convert numpy types to Python native types for JSON serialization.
    
    Parameters:
    -----------
    obj : any
        Object to convert
        
    Returns:
    --------
    any
        Object with numpy types converted to Python native types
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


# Dictionary mapping clustering method names to their implementations
CLUSTERING_METHODS = {
    'agglomerative': 'clustering_agglomerative',
    'pca_agglomerative': 'clustering_pca_agglomerative',
    'gmm': 'clustering_gmm',
    'pca_gmm': 'clustering_pca_gmm',
    'spherical_kmeans': 'clustering_spherical_kmeans',
    'pca_kmeans': 'clustering_pca_kmeans',
    'sae_topk': 'clustering_sae_topk'
}

# Set of supported clustering methods
SUPPORTED_CLUSTERING_METHODS = {
    'agglomerative',
    'pca_agglomerative',
    'gmm',
    'pca_gmm',
    'spherical_kmeans',
    'pca_kmeans',
    'sae_topk'
} 