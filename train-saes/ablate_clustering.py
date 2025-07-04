# %%
import os
import numpy as np
import torch
import argparse
import json
from sklearn.cluster import KMeans, AgglomerativeClustering

from tqdm import tqdm
import random
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from utils import utils
import gc
import time
from utils.clustering import (
    print_and_flush,
    compute_centroid_orthogonality,
    save_clustering_model,
    compute_silhouette_score
)

# %%

parser = argparse.ArgumentParser(description="K-means clustering and autograding of neural activations")
parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
                    help="Model to analyze")
parser.add_argument("--layer", type=int, default=12,
                    help="Layer to analyze")
parser.add_argument("--n_examples", type=int, default=500,
                    help="Number of examples to analyze")
parser.add_argument("--min_clusters", type=int, default=4,
                    help="Minimum number of clusters")
parser.add_argument("--max_clusters", type=int, default=20,
                    help="Maximum number of clusters")
parser.add_argument("--load_in_8bit", action="store_true", default=False,
                    help="Load model in 8-bit mode")
parser.add_argument("--n_autograder_examples", type=int, default=100,
                    help="Number of examples from each cluster to use for autograding")
parser.add_argument("--description_examples", type=int, default=50,
                    help="Number of examples to use for generating cluster descriptions")
parser.add_argument("--clustering_methods", type=str, nargs='+', 
                    default=["gmm", "pca_gmm", "spherical_kmeans", "pca_kmeans", "agglomerative", "pca_agglomerative", "sae_topk"],
                    help="Clustering methods to use")
parser.add_argument("--clustering_pilot_size", type=int, default=50_000,
                    help="Number of samples to use for pilot fitting with GMM")
parser.add_argument("--clustering_pilot_n_init", type=int, default=10,
                    help="Number of initializations for pilot fitting with GMM")
parser.add_argument("--clustering_pilot_max_iter", type=int, default=100,
                    help="Maximum iterations for pilot fitting with GMM")
parser.add_argument("--clustering_full_n_init", type=int, default=1,
                    help="Number of initializations for full fitting with GMM")
parser.add_argument("--clustering_full_max_iter", type=int, default=100,
                    help="Maximum iterations for full fitting with GMM")
parser.add_argument("--silhouette_sample_size", type=int, default=50_000,
                    help="Number of samples to use for silhouette score calculation")
args, _ = parser.parse_known_args()

# %%
def clustering_agglomerative(example_activations, n_clusters, args):
    """
    Perform Agglomerative Hierarchical clustering on normalized activations.
    
    Parameters:
    -----------
    example_activations : numpy.ndarray
        Normalized activation vectors
    n_clusters : int
        Number of clusters
    args : argparse.Namespace, optional
        Command line arguments (not used in this method)
        
    Returns:
    --------
    tuple
        (cluster_labels, cluster_centers, silhouette)
    """
    # Initialize Agglomerative Clustering with Ward linkage
    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric='cosine',
        linkage='average'
    )
    
    # Fit model
    cluster_labels = model.fit_predict(example_activations)
    
    # Compute cluster centers (not provided by the model)
    cluster_centers = np.zeros((n_clusters, example_activations.shape[1]))
    for i in range(n_clusters):
        mask = cluster_labels == i
        if np.any(mask):
            cluster_centers[i] = np.mean(example_activations[mask], axis=0)
    
    # Calculate silhouette score
    silhouette = compute_silhouette_score(example_activations, cluster_labels, sample_size=args.silhouette_sample_size, random_state=42)
    
    # Save the clustering model
    model_id = args.model.split('/')[-1].lower()
    clustering_data = {
        'model': model,
        'cluster_labels': cluster_labels,
        'cluster_centers': cluster_centers,
        'silhouette': silhouette,
        'n_clusters': n_clusters,
        'input_dim': example_activations.shape[1],
        'method': 'agglomerative'
    }
    
    save_clustering_model(clustering_data, model_id, args.layer, n_clusters, 'agglomerative')
    
    return cluster_labels, cluster_centers, silhouette

def clustering_spherical_kmeans(example_activations, n_clusters, args):
    """
    Perform Spherical KMeans clustering using cosine similarity.
    
    Parameters:
    -----------
    example_activations : numpy.ndarray
        Normalized activation vectors
    n_clusters : int
        Number of clusters
    args : argparse.Namespace, optional
        Command line arguments (not used in this method)
        
    Returns:
    --------
    tuple
        (cluster_labels, cluster_centers, silhouette)
    """
    start_time = time.time()
    # Initialize KMeans
    kmeans = KMeans(
        n_clusters=n_clusters,
        n_init='auto',
        random_state=42,
        verbose=1
    )

    activations_norm = example_activations / np.linalg.norm(example_activations, axis=1, keepdims=True)
    
    # Fit model on normalized data (for cosine similarity)
    cluster_labels = kmeans.fit_predict(activations_norm)
    
    # Get the cluster centers
    cluster_centers = kmeans.cluster_centers_
    
    # Normalize cluster centers for cosine similarity
    norms = np.linalg.norm(cluster_centers, axis=1, keepdims=True)
    cluster_centers = cluster_centers / norms
    
    # Calculate silhouette score using cosine distance
    silhouette = compute_silhouette_score(activations_norm, cluster_labels, sample_size=args.silhouette_sample_size, random_state=42)
    
    # Save the clustering model
    model_id = args.model.split('/')[-1].lower()
    clustering_data = {
        'model': kmeans,
        'cluster_labels': cluster_labels,
        'cluster_centers': cluster_centers,
        'silhouette': silhouette,
        'n_clusters': n_clusters,
        'input_dim': example_activations.shape[1],
        'method': 'spherical_kmeans',
        'activations_norm': activations_norm
    }
    
    save_clustering_model(clustering_data, model_id, args.layer, n_clusters, 'spherical_kmeans')
    
    print_and_flush(f"    Spherical KMeans clustering completed in {time.time() - start_time:.2f} seconds total")

    return cluster_labels, cluster_centers, silhouette

def clustering_gmm(example_activations, n_clusters, args):
    """
    Perform Gaussian Mixture Model clustering with pilot-fit then fine-tune approach.
    
    Parameters:
    -----------
    example_activations : numpy.ndarray
        Normalized activation vectors
    n_clusters : int
        Number of clusters
    args : argparse.Namespace, optional
        Command line arguments containing clustering parameters
        
    Returns:
    --------
    tuple
        (cluster_labels, cluster_centers, silhouette)
    """
    start_time = time.time()
    n_samples = example_activations.shape[0]
    
    # Use clustering parameters from args
    pilot_size = min(args.clustering_pilot_size, n_samples)
    pilot_n_init = args.clustering_pilot_n_init
    pilot_max_iter = args.clustering_pilot_max_iter
    full_n_init = args.clustering_full_n_init
    full_max_iter = args.clustering_full_max_iter

    print_and_flush(f"    GMM clustering with {n_clusters} clusters on {n_samples} samples...")
    
    if n_samples > pilot_size:
        # Step 1: Pilot fit on sub-sample
        print_and_flush(f"    Step 1: Pilot fit on {pilot_size} samples with n_init={pilot_n_init}, max_iter={pilot_max_iter}...")
        pilot_start = time.time()
        
        # Sub-sample
        pilot_indices = np.random.choice(n_samples, pilot_size, replace=False)
        pilot_data = example_activations[pilot_indices]
        
        # Pilot GMM with multiple initializations
        pilot_gmm = GaussianMixture(
            n_components=n_clusters,
            covariance_type='diag',
            random_state=42,
            n_init=pilot_n_init,
            max_iter=pilot_max_iter,
            verbose=1
        )
        pilot_gmm.fit(pilot_data)
        pilot_time = time.time() - pilot_start
        print_and_flush(f"    Pilot fit completed in {pilot_time:.2f} seconds")
        
        # Step 2: Fine-tune on full dataset with good initialization
        print_and_flush(f"    Step 2: Fine-tune on full {n_samples} samples with n_init={full_n_init}, max_iter={full_max_iter}...")
        finetune_start = time.time()
        
        # Create final GMM with initialization from pilot
        gmm = GaussianMixture(
            n_components=n_clusters,
            covariance_type='diag',
            random_state=42,
            n_init=full_n_init,
            max_iter=full_max_iter,
            weights_init=pilot_gmm.weights_,
            means_init=pilot_gmm.means_,
            precisions_init=pilot_gmm.precisions_,
            verbose=1
        )
        gmm.fit(example_activations)
        finetune_time = time.time() - finetune_start
        print_and_flush(f"    Fine-tune completed in {finetune_time:.2f} seconds")
        
    else:
        # For small datasets, use standard approach
        print_and_flush(f"    Small dataset ({n_samples} samples), using standard approach with n_init=10...")
        gmm = GaussianMixture(
            n_components=n_clusters,
            covariance_type='diag',
            random_state=42,
            n_init=10,
            verbose=1
        )
        gmm.fit(example_activations)
    
    # Get cluster assignments
    cluster_labels = gmm.predict(example_activations)
    
    # Use means as cluster centers
    cluster_centers = gmm.means_
    
    # Calculate silhouette score
    silhouette = compute_silhouette_score(example_activations, cluster_labels, sample_size=args.silhouette_sample_size, random_state=42)
    
    # Save the clustering model
    model_id = args.model.split('/')[-1].lower()
    clustering_data = {
        'model': gmm,
        'cluster_labels': cluster_labels,
        'cluster_centers': cluster_centers,
        'silhouette': silhouette,
        'n_clusters': n_clusters,
        'input_dim': example_activations.shape[1],
        'method': 'gmm'
    }
    
    save_clustering_model(clustering_data, model_id, args.layer, n_clusters, 'gmm')
    
    total_time = time.time() - start_time
    print_and_flush(f"    GMM clustering completed in {total_time:.2f} seconds total")
    
    return cluster_labels, cluster_centers, silhouette

def clustering_pca_kmeans(example_activations, n_clusters, args):
    """
    Perform PCA dimensionality reduction followed by KMeans clustering.
    
    Parameters:
    -----------
    example_activations : numpy.ndarray
        Normalized activation vectors
    n_clusters : int
        Number of clusters
    args : argparse.Namespace, optional
        Command line arguments (not used in this method)
        
    Returns:
    --------
    tuple
        (cluster_labels, cluster_centers, silhouette)
    """
    start_time = time.time()
    # Determine number of PCA components (min of n_samples, n_features, 100)
    n_components = min(example_activations.shape[0], example_activations.shape[1], 100)
    
    # Apply PCA
    pca = PCA(n_components=n_components, random_state=42)
    reduced_data = pca.fit_transform(example_activations)
    
    # Apply KMeans to reduced data
    kmeans = KMeans(
        n_clusters=n_clusters,
        n_init='auto',
        random_state=42,
        verbose=1
    )
    cluster_labels = kmeans.fit_predict(reduced_data)
    
    # Compute cluster centers in original space
    cluster_centers = np.zeros((n_clusters, example_activations.shape[1]))
    for i in range(n_clusters):
        mask = cluster_labels == i
        if np.any(mask):
            cluster_centers[i] = np.mean(example_activations[mask], axis=0)
    
    # Calculate silhouette score in reduced space for efficiency
    silhouette = compute_silhouette_score(reduced_data, cluster_labels, sample_size=args.silhouette_sample_size, random_state=42)
    
    # Save the clustering model
    model_id = args.model.split('/')[-1].lower()
    clustering_data = {
        'model': kmeans,
        'pca': pca,
        'cluster_labels': cluster_labels,
        'cluster_centers': cluster_centers,
        'silhouette': silhouette,
        'n_clusters': n_clusters,
        'input_dim': example_activations.shape[1],
        'method': 'pca_kmeans',
        'reduced_data': reduced_data
    }
    
    save_clustering_model(clustering_data, model_id, args.layer, n_clusters, 'pca_kmeans')
    
    print_and_flush(f"    PCA+KMeans clustering completed in {time.time() - start_time:.2f} seconds total")
    
    return cluster_labels, cluster_centers, silhouette

def clustering_pca_gmm(example_activations, n_clusters, args):
    """
    Perform PCA dimensionality reduction followed by GMM clustering with pilot-fit then fine-tune approach.
    
    Parameters:
    -----------
    example_activations : numpy.ndarray
        Normalized activation vectors
    n_clusters : int
        Number of clusters
    args : argparse.Namespace, optional
        Command line arguments containing clustering parameters
        
    Returns:
    --------
    tuple
        (cluster_labels, cluster_centers, silhouette)
    """
    start_time = time.time()
    n_samples = example_activations.shape[0]
    
    # Use clustering parameters from args
    pilot_size = min(args.clustering_pilot_size, n_samples)
    pilot_n_init = args.clustering_pilot_n_init
    pilot_max_iter = args.clustering_pilot_max_iter
    full_n_init = args.clustering_full_n_init
    full_max_iter = args.clustering_full_max_iter
    
    print_and_flush(f"    PCA+GMM clustering with {n_clusters} clusters on {n_samples} samples...")
    
    # Determine number of PCA components (min of n_samples, n_features, 100)
    n_components = min(example_activations.shape[0], example_activations.shape[1], 100)
    
    # Apply PCA
    print_and_flush(f"    Applying PCA to {n_components} components...")
    pca_start = time.time()
    pca = PCA(n_components=n_components, random_state=42)
    reduced_data = pca.fit_transform(example_activations)
    pca_time = time.time() - pca_start
    print_and_flush(f"    PCA completed in {pca_time:.2f} seconds")
    
    if n_samples > pilot_size:
        # Step 1: Pilot fit on sub-sample
        print_and_flush(f"    Step 1: Pilot fit on {pilot_size} samples with n_init={pilot_n_init}, max_iter={pilot_max_iter}...")
        pilot_start = time.time()
        
        # Sub-sample with fixed random state for reproducibility
        np.random.seed(42)
        pilot_indices = np.random.choice(n_samples, pilot_size, replace=False)
        pilot_reduced_data = reduced_data[pilot_indices]
        
        # Pilot GMM with multiple initializations
        pilot_gmm = GaussianMixture(
            n_components=n_clusters,
            covariance_type='full',
            random_state=42,
            n_init=pilot_n_init,
            max_iter=pilot_max_iter,
            verbose=1
        )
        pilot_gmm.fit(pilot_reduced_data)
        pilot_time = time.time() - pilot_start
        print_and_flush(f"    Pilot fit completed in {pilot_time:.2f} seconds")
        
        # Step 2: Fine-tune on full dataset with good initialization
        print_and_flush(f"    Step 2: Fine-tune on full {n_samples} samples with n_init={full_n_init}, max_iter={full_max_iter}...")
        finetune_start = time.time()
        
        # Create final GMM with initialization from pilot
        gmm = GaussianMixture(
            n_components=n_clusters,
            covariance_type='full',
            random_state=42,
            n_init=full_n_init,
            max_iter=full_max_iter,
            weights_init=pilot_gmm.weights_,
            means_init=pilot_gmm.means_,
            precisions_init=pilot_gmm.precisions_,
            verbose=1
        )
        gmm.fit(reduced_data)
        finetune_time = time.time() - finetune_start
        print_and_flush(f"    Fine-tune completed in {finetune_time:.2f} seconds")
        
    else:
        # For small datasets, use standard approach
        print_and_flush(f"    Small dataset ({n_samples} samples), using standard approach with n_init=10...")
        gmm = GaussianMixture(
            n_components=n_clusters,
            covariance_type='full',
            random_state=42,
            n_init=10
        )
        gmm.fit(reduced_data)
    
    # Get cluster assignments
    cluster_labels = gmm.predict(reduced_data)
    
    # Compute cluster centers in original space
    cluster_centers = np.zeros((n_clusters, example_activations.shape[1]))
    for i in range(n_clusters):
        mask = cluster_labels == i
        if np.any(mask):
            cluster_centers[i] = np.mean(example_activations[mask], axis=0)
    
    # Calculate silhouette score in reduced space for efficiency
    silhouette = compute_silhouette_score(reduced_data, cluster_labels, sample_size=args.silhouette_sample_size, random_state=42)
    
    # Save the clustering model
    model_id = args.model.split('/')[-1].lower()
    clustering_data = {
        'model': gmm,
        'pca': pca,
        'cluster_labels': cluster_labels,
        'cluster_centers': cluster_centers,
        'silhouette': silhouette,
        'n_clusters': n_clusters,
        'input_dim': example_activations.shape[1],
        'method': 'pca_gmm',
        'reduced_data': reduced_data
    }
    
    save_clustering_model(clustering_data, model_id, args.layer, n_clusters, 'pca_gmm')
    
    total_time = time.time() - start_time
    print_and_flush(f"    PCA+GMM clustering completed in {total_time:.2f} seconds total")
    
    return cluster_labels, cluster_centers, silhouette

def clustering_pca_agglomerative(example_activations, n_clusters, args):
    """
    Perform PCA dimensionality reduction followed by Agglomerative clustering.
    
    Parameters:
    -----------
    example_activations : numpy.ndarray
        Normalized activation vectors
    n_clusters : int
        Number of clusters
    args : argparse.Namespace, optional
        Command line arguments (not used in this method)
        
    Returns:
    --------
    tuple
        (cluster_labels, cluster_centers, silhouette)
    """
    start_time = time.time()
    # Determine number of PCA components (min of n_samples, n_features, 100)
    n_components = min(example_activations.shape[0], example_activations.shape[1], 100)
    
    # Apply PCA
    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(example_activations)
    
    # Apply Agglomerative clustering to reduced data
    agg = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric='cosine',
        linkage='average'
    )
    cluster_labels = agg.fit_predict(reduced_data)
    
    # Compute cluster centers in original space
    cluster_centers = np.zeros((n_clusters, example_activations.shape[1]))
    for i in range(n_clusters):
        mask = cluster_labels == i
        if np.any(mask):
            cluster_centers[i] = np.mean(example_activations[mask], axis=0)
    
    # Calculate silhouette score in reduced space for efficiency
    silhouette = compute_silhouette_score(reduced_data, cluster_labels, sample_size=args.silhouette_sample_size, random_state=42)
    
    # Save the clustering model
    model_id = args.model.split('/')[-1].lower()
    clustering_data = {
        'model': agg,
        'pca': pca,
        'cluster_labels': cluster_labels,
        'cluster_centers': cluster_centers,
        'silhouette': silhouette,
        'n_clusters': n_clusters,
        'input_dim': example_activations.shape[1],
        'method': 'pca_agglomerative',
        'reduced_data': reduced_data
    }
    
    save_clustering_model(clustering_data, model_id, args.layer, n_clusters, 'pca_agglomerative')
    
    print_and_flush(f"    PCA+Agglomerative clustering completed in {time.time() - start_time:.2f} seconds total")
    
    return cluster_labels, cluster_centers, silhouette

 
def clustering_sae_topk(example_activations, n_clusters, args, topk=3):
    """
    Perform clustering using a top-k sparse autoencoder.
    Follows the TinySAE implementation from https://github.com/JoshEngels/TinySAE/blob/main/tiny_sae.py
    
    Parameters:
    -----------
    example_activations : numpy.ndarray
        Normalized activation vectors
    n_clusters : int
        Number of clusters (also number of features in the SAE)
    args : argparse.Namespace, optional
        Command line arguments (not used in this method)
    topk : int
        Number of top activations to keep during training (default: 3)
        
    Returns:
    --------
    tuple
        (cluster_labels, cluster_centers, silhouette)
    """
    start_time = time.time()
    # Ensure we're working with torch tensors on the appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Convert numpy array to torch tensor
    X = torch.from_numpy(example_activations).float().to(device)
    input_dim = example_activations.shape[1]
    
    # Initialize model, loss, and optimizer
    sae = utils.SAE(input_dim, n_clusters, k=topk).to(device)
    # Auto-select LR using 1 / sqrt(d) scaling law from TinySAE
    lr = 2e-4 / (n_clusters / (2**14)) ** 0.5
    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)
    
    # Train the autoencoder
    max_epochs = 300
    batch_size = min(512, example_activations.shape[0])  # Adjust batch size based on data size
    n_samples = X.shape[0]
    
    # Early stopping parameters
    patience = 10
    best_loss = float('inf')
    patience_counter = 0
    
    print_and_flush(f"Training sparse autoencoder with {n_clusters} clusters, topk={topk}...")
    for epoch in range(max_epochs):
        # Shuffle data
        indices = torch.randperm(n_samples)
        
        # Set model to training mode
        sae.train()
        total_loss = 0
        
        # Create mini-batches and train
        for i in range(0, n_samples, batch_size):
            batch_indices = indices[i:min(i+batch_size, n_samples)]
            batch_X = X[batch_indices]
            
            # Forward pass
            predicted = sae(batch_X)
            
            # Compute loss - using the Loss function from TinySAE
            error = predicted - batch_X
            loss = (error**2).sum()
            loss /= ((batch_X - batch_X.mean(dim=0, keepdim=True)) ** 2).sum()
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Apply decoder normalization after each step (as in TinySAE)
            sae.set_decoder_norm_to_unit_norm()
            
            total_loss += loss.item() * len(batch_indices)
            
            # Free up memory
            del batch_X, predicted, error, loss
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Calculate average loss
        avg_loss = total_loss / n_samples
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print_and_flush(f"Epoch [{epoch+1}/{max_epochs}], Loss: {avg_loss:.6f}")
        
        # Early stopping check
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            
            # Save best model state
            best_model_state = {
                'encoder_weight': sae.encoder.weight.data.clone(),
                'encoder_bias': sae.encoder.bias.data.clone(),
                'decoder_weight': sae.W_dec.data.clone(),
                'b_dec': sae.b_dec.data.clone()
            }
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print_and_flush(f"Early stopping at epoch {epoch+1}")
                
                # Restore best model
                sae.encoder.weight.data = best_model_state['encoder_weight']
                sae.encoder.bias.data = best_model_state['encoder_bias']
                sae.W_dec.data = best_model_state['decoder_weight']
                sae.b_dec.data = best_model_state['b_dec']
                    
                break
    
    # Create directory for saving SAE models
    os.makedirs('results/vars/clustering_models', exist_ok=True)
    
    # Save the SAE model
    # Get model_id from args
    model_id = args.model.split('/')[-1].lower()
    sae_save_path = f'results/vars/clustering_models/sae_{model_id}_layer{args.layer}_clusters{n_clusters}.pt'
    torch.save({
        'encoder_weight': sae.encoder.weight.data.clone().cpu(),
        'encoder_bias': sae.encoder.bias.data.clone().cpu(),
        'decoder_weight': sae.W_dec.data.clone().cpu(),
        'b_dec': sae.b_dec.data.clone().cpu(),
        'input_dim': input_dim,
        'num_latents': n_clusters,
        'topk': topk,
        'loss': best_loss
    }, sae_save_path)
    print_and_flush(f"Saved SAE model to {sae_save_path}")
    
    # Use the encoder to determine cluster assignments - get the highest activating feature for each example
    sae.eval()
    with torch.no_grad():
        # Process in batches to avoid memory issues with large datasets
        all_top_features = []
        for i in range(0, n_samples, batch_size):
            batch_X = X[i:min(i+batch_size, n_samples)]
            # Get the full activations without topk restriction for final cluster assignment
            activations = sae.encoder(batch_X - sae.b_dec)
            # Get the index of the maximum activation for each example
            top_feature = activations.argmax(dim=1)
            all_top_features.append(top_feature.cpu())
            
            # Free memory
            del batch_X, activations, top_feature
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        # Combine all batches
        if len(all_top_features) > 1:
            cluster_labels = torch.cat(all_top_features, dim=0).numpy()
        else:
            cluster_labels = all_top_features[0].numpy()
    
    # Get cluster centers (decoder weights)
    # Each row of the decoder weight matrix corresponds to a cluster centroid
    cluster_centers = sae.W_dec.data.cpu().numpy()
    
    # Normalize cluster centers
    norms = np.linalg.norm(cluster_centers, axis=1, keepdims=True)
    cluster_centers = cluster_centers / (norms + 1e-8)  # Add small epsilon to avoid division by zero
    
    # Calculate silhouette score
    silhouette = compute_silhouette_score(example_activations, cluster_labels, sample_size=args.silhouette_sample_size, random_state=42)
    
    clustering_data = {
        'model': None,  # SAE models are saved separately, so we don't need to save the model here
        'cluster_labels': cluster_labels,
        'cluster_centers': cluster_centers,
        'silhouette': silhouette,
        'n_clusters': n_clusters,
        'input_dim': example_activations.shape[1],
        'method': 'sae_topk',
        'sae_path': sae_save_path  # Store path to SAE model
    }
    
    save_clustering_model(clustering_data, model_id, args.layer, n_clusters, 'sae_topk')
    
    # Clean up to free memory
    del sae, X
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    print_and_flush(f"    Sparse autoencoder clustering completed in {time.time() - start_time:.2f} seconds total. Silhouette score: {silhouette:.4f}")
    
    return cluster_labels, cluster_centers, silhouette

# Dictionary mapping clustering method names to their implementations
CLUSTERING_METHODS = {
    'agglomerative': clustering_agglomerative,
    'pca_agglomerative': clustering_pca_agglomerative,
    'gmm': clustering_gmm,
    'pca_gmm': clustering_pca_gmm,
    'spherical_kmeans': clustering_spherical_kmeans,
    'pca_kmeans': clustering_pca_kmeans,
    'sae_topk': clustering_sae_topk
}

def generate_representative_examples(cluster_centers, texts, cluster_labels, example_activations):
    """
    Generate representative examples for each cluster based on distance to centroid.
    
    Parameters:
    -----------
    cluster_centers : numpy.ndarray
        Cluster centers
    texts : list
        List of texts
    cluster_labels : numpy.ndarray
        Cluster labels for each text
    example_activations : numpy.ndarray
        Normalized activation vectors
        
    Returns:
    --------
    dict
        Dictionary mapping cluster_idx to list of representative examples
    """
    representative_examples = {}
    
    for cluster_idx in range(len(cluster_centers)):
        # Get indices of texts in this cluster
        cluster_indices = np.where(cluster_labels == cluster_idx)[0]
        
        # Skip empty clusters
        if len(cluster_indices) == 0:
            representative_examples[cluster_idx] = []
            continue
            
        # Get all examples in this cluster
        cluster_texts = [texts[i] for i in cluster_indices]
        
        # Calculate distances to centroid
        cluster_vectors = np.stack([example_activations[i] for i in cluster_indices])
        centroid = cluster_centers[cluster_idx]
        distances = np.linalg.norm(cluster_vectors - centroid, axis=1)
        
        # Sort examples by distance to centroid
        sorted_indices = np.argsort(distances)
        sorted_examples = [cluster_texts[i] for i in sorted_indices]
        
        representative_examples[cluster_idx] = sorted_examples
    
    return representative_examples

def generate_category_descriptions(cluster_centers, texts, cluster_labels, example_activations, model_name, n_description_examples=5):
    """
    Generate descriptions for each cluster based on most representative sentences.
    Uses half top examples and half random examples from the cluster.
    
    Parameters:
    -----------
    cluster_centers : numpy.ndarray
        Cluster centers
    texts : list
        List of texts
    cluster_labels : numpy.ndarray
        Cluster labels for each text
    example_activations : numpy.ndarray
        Normalized activation vectors
    model_name : str
        Name of the model to use for generating descriptions
    n_description_examples : int
        Number of examples to use for generating descriptions
        
    Returns:
    --------
    list
        List of tuples (cluster_id, category_title, category_description)
    """
    categories = []
    representative_examples = generate_representative_examples(
        cluster_centers, texts, cluster_labels, example_activations
    )
    
    for cluster_idx in range(len(cluster_centers)):
        # Skip empty clusters
        if len(representative_examples[cluster_idx]) == 0:
            continue

        # Get top examples
        examples = representative_examples[cluster_idx][:n_description_examples]
    
        # Generate title and description
        for _ in range(3):
            try:
                title, description = utils.generate_cluster_description(examples, model_name=model_name, n_trace_examples=3)
                categories.append((str(cluster_idx), title, description))
                break
            except Exception as e:
                # Fallback to simple description
                print_and_flush(f"Error generating description for cluster {cluster_idx}: {e}")
                time.sleep(5)
    
    return categories

def evaluate_clustering_accuracy(texts, cluster_labels, categories, n_autograder_examples=5):
    """
    Evaluate clustering using the binary accuracy autograder.
    Tests each cluster independently against examples from other clusters.
    
    Parameters:
    -----------
    texts : list
        List of texts
    cluster_labels : numpy.ndarray
        Cluster labels for each text
    categories : list
        List of tuples (cluster_id, title, description)
    n_autograder_examples : int
        Number of examples from each cluster to use for autograding
        
    Returns:
    --------
    dict
        Autograder results including precision, recall, accuracy and F1 for each cluster
    """
    # Convert cluster_labels to list of strings for compatibility
    str_cluster_labels = [str(label) for label in cluster_labels]
    
    # Run binary autograder
    for _ in range(3):
        try:
            results = utils.accuracy_autograder(texts, categories, str_cluster_labels, 
                                               n_autograder_examples=n_autograder_examples)
            break
        except Exception as e:
            print_and_flush(f"Error running accuracy autograder: {e}")
            time.sleep(5)
    
    print_and_flush(results["avg"])
    return results

def evaluate_clustering_completeness(texts, categories, n_test_examples=50):
    """
    Evaluate clustering using the completeness autograder with a random sample of texts.
    
    Parameters:
    -----------
    texts : list
        List of texts
    categories : list
        List of tuples (cluster_id, title, description)
    n_test_examples : int
        Number of examples to use for testing completeness
        
    Returns:
    --------
    dict
        Autograder results
    """
    # Sample n_test_examples randomly from all texts
    if len(texts) > n_test_examples:
        test_texts = random.sample(texts, n_test_examples)
    else:
        test_texts = texts
    
    # Run autograder on the sampled texts
    for _ in range(3):
        try:
            results = utils.completeness_autograder(test_texts, categories)
            break
        except Exception as e:
            print_and_flush(f"Error running completeness autograder: {e}")
            time.sleep(5)
    
    return results


def evaluate_clustering(texts, cluster_labels, n_clusters, example_activations, cluster_centers, 
                       model_name, n_autograder_examples=5, n_description_examples=5):
    """
    Evaluate clustering using both accuracy and optionally completeness autograders.
    
    Parameters:
    -----------
    texts : list
        List of texts
    cluster_labels : numpy.ndarray
        Cluster labels for each text
    n_clusters : int
        Number of clusters
    example_activations : numpy.ndarray
        Normalized activation vectors
    cluster_centers : numpy.ndarray
        Cluster centers
    model_name : str
        Name of the model to use for generating descriptions
    n_autograder_examples : int
        Number of examples from each cluster to use for autograding
    n_description_examples : int
        Number of examples to use for generating descriptions
        
    Returns:
    --------
    dict
        Combined evaluation results
    """
    # Generate category descriptions
    categories = generate_category_descriptions(
        cluster_centers, texts, cluster_labels, example_activations, model_name, n_description_examples
    )
    
    # Run binary accuracy autograder (evaluates each cluster independently)
    accuracy_results = evaluate_clustering_accuracy(
        texts, cluster_labels, categories, n_autograder_examples
    )
    
    # Compute centroid orthogonality
    orthogonality = compute_centroid_orthogonality(cluster_centers)
    
    # Get average accuracy from accuracy_results["avg"]
    avg_accuracy = accuracy_results.get("avg", {}).get("accuracy", 0)
    
    results = {
        "accuracy": avg_accuracy,
        "categories": categories,
        "orthogonality": orthogonality  # Add orthogonality to results
    }
    
    # Optionally run completeness autograder
    completeness_results = evaluate_clustering_completeness(texts, categories)
    results["assigned_fraction"] = completeness_results["assigned_fraction"]
    results["category_counts"] = completeness_results["category_counts"]

    # Create detailed results by cluster
    detailed_results = {}
    for cluster_id, title, description in categories:
        cluster_id_str = str(cluster_id)
        cluster_metrics = accuracy_results.get(cluster_id_str, {})
        cluster_idx = int(cluster_id)
        cluster_examples = generate_representative_examples(
            cluster_centers, texts, cluster_labels, example_activations
        )[cluster_idx]
        
        detailed_results[cluster_id_str] = {
            'title': title,
            'description': description,
            'size': len(np.where(cluster_labels == cluster_idx)[0]),
            'precision': cluster_metrics.get('precision', 0),
            'recall': cluster_metrics.get('recall', 0),
            'accuracy': cluster_metrics.get('accuracy', 0),
            'f1': cluster_metrics.get('f1', 0),
            'examples': cluster_examples[:15]  # Top 15 examples
        }
    
    results['detailed_results'] = detailed_results
    
    return results


def run_clustering_experiment(clustering_method, clustering_func, all_texts, example_activations, args, model_id=None):
    """
    Run a clustering experiment using the specified clustering method.
    
    Parameters:
    -----------
    clustering_method : str
        Name of the clustering method
    clustering_func : function
        Function that implements the clustering algorithm
    all_texts : list
        List of texts to cluster
    example_activations : numpy.ndarray
        Normalized activation vectors
    args : argparse.Namespace
        Command line arguments
    model_id : str
        Model identifier for file naming
        
    Returns:
    --------
    dict
        Results of the clustering experiment
    """
    print_and_flush(f"\nRunning {clustering_method.upper()} clustering experiment...")
    
    # For methods that require n_clusters, use the original code
    silhouette_scores = []
    accuracy_scores = []
    f1_scores = []
    assignment_rates = []
    orthogonality_scores = []  # Add orthogonality scores
    precision_scores = []  # Add precision scores
    recall_scores = []  # Add recall scores
    detailed_results_dict = {}
    
    cluster_range = list(range(args.min_clusters, args.max_clusters + 1))
    
    print_and_flush(f"Testing {len(cluster_range)} different cluster counts...")
    for n_clusters in tqdm(cluster_range, desc=f"{clustering_method.capitalize()} progress"):
        # Perform clustering
        cluster_labels, cluster_centers, silhouette = clustering_func(example_activations, n_clusters, args)
        
        # Evaluate clustering
        evaluation_results = evaluate_clustering(
            all_texts, 
            cluster_labels, 
            n_clusters, 
            example_activations,
            cluster_centers,
            args.model,
            args.n_autograder_examples,
            args.description_examples,
        )
        
        # Store metrics
        silhouette_scores.append(silhouette)
        accuracy_scores.append(evaluation_results['accuracy'])
        orthogonality_scores.append(evaluation_results['orthogonality'])
        
        # Calculate average F1 score, precision, and recall across all clusters
        f1_sum = 0.0
        precision_sum = 0.0
        recall_sum = 0.0
        f1_count = 0
        for cluster_id, metrics in evaluation_results['detailed_results'].items():
            if metrics['f1'] > 0:  # Only count non-zero F1 scores
                f1_sum += metrics['f1']
                precision_sum += metrics['precision']
                recall_sum += metrics['recall']
                f1_count += 1
        avg_f1 = f1_sum / f1_count if f1_count > 0 else 0
        avg_precision = precision_sum / f1_count if f1_count > 0 else 0
        avg_recall = recall_sum / f1_count if f1_count > 0 else 0
        
        f1_scores.append(avg_f1)
        precision_scores.append(avg_precision)
        recall_scores.append(avg_recall)
        
        # Store assignment rate if completeness was run
        assignment_rates.append(evaluation_results.get('assigned_fraction', 0))
        
        # Store detailed results
        detailed_results_dict[n_clusters] = evaluation_results

    # Identify optimal number of clusters based on accuracy only
    optimal_n_clusters = cluster_range[np.argmax(accuracy_scores)]

    # Create a concise results JSON
    results_data = {
        "clustering_method": clustering_method,
        "model_id": model_id,
        "layer": args.layer,
        "cluster_range": cluster_range,
        "silhouette_scores": silhouette_scores,
        "accuracy_scores": accuracy_scores,
        "precision_scores": precision_scores,
        "recall_scores": recall_scores,
        "f1_scores": f1_scores,
        "assignment_rates": assignment_rates,
        "orthogonality_scores": orthogonality_scores,
        "optimal_n_clusters": optimal_n_clusters,
        "optimal_silhouette": silhouette_scores[cluster_range.index(optimal_n_clusters)],
        "optimal_accuracy": accuracy_scores[cluster_range.index(optimal_n_clusters)],
        "optimal_precision": precision_scores[cluster_range.index(optimal_n_clusters)],
        "optimal_recall": recall_scores[cluster_range.index(optimal_n_clusters)],
        "optimal_f1": f1_scores[cluster_range.index(optimal_n_clusters)],
        "optimal_assignment_rate": assignment_rates[cluster_range.index(optimal_n_clusters)],
        "optimal_orthogonality": orthogonality_scores[cluster_range.index(optimal_n_clusters)],
        "detailed_results": detailed_results_dict
    }

    # Convert any numpy types to Python native types for JSON serialization
    results_data = utils.convert_numpy_types(results_data)
    
    # Save results to JSON
    results_json_path = f'results/vars/{clustering_method}_results_{model_id}_layer{args.layer}.json'
    with open(results_json_path, 'w') as f:
        json.dump(results_data, f, indent=2, cls=utils.NumpyEncoder)
    print_and_flush(f"Saved {clustering_method} results to {results_json_path}")
    
    return results_data


# %% Load model and process activations
print_and_flush("Loading model and processing activations...")
model, tokenizer = utils.load_model(
    model_name=args.model,
    load_in_8bit=args.load_in_8bit
)

# %% Get model identifier for file naming
model_id = args.model.split('/')[-1].lower()

# %% Process saved responses
all_activations, all_texts, overall_mean = utils.process_saved_responses(
    args.model, 
    args.n_examples,
    model,
    tokenizer,
    args.layer
)

del model, tokenizer
torch.cuda.empty_cache()
gc.collect()

# %% Center activations
all_activations = [x - overall_mean for x in all_activations]
all_activations = np.stack([a.cpu().numpy().reshape(-1) for a in all_activations])
norms = np.linalg.norm(all_activations, axis=1, keepdims=True)
all_activations = all_activations / norms

# %% Filter clustering methods based on args
clustering_methods = [method for method in args.clustering_methods if method in CLUSTERING_METHODS]

# Run each clustering method
current_results = {}
for method in clustering_methods:
    try:
        clustering_func = CLUSTERING_METHODS[method]
        results = run_clustering_experiment(method, clustering_func, all_texts, all_activations, args, model_id)
        current_results[method] = results
    except Exception as e:
        print_and_flush(f"Error running {method}: {e}")
