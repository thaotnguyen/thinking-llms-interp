# %%
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import math
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
import re
import matplotlib.gridspec as gridspec
import matplotlib as mpl
import argparse
    
parser = argparse.ArgumentParser(description="Visualize vector losses")
parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B",
                    help="Model name")
parser.add_argument("--smoothing_sigma", type=float, default=5.0,
                    help="Sigma parameter for Gaussian smoothing")

args, _ = parser.parse_known_args()

def smooth_data(data, sigma=2):
    """
    Apply Gaussian smoothing to data.
    
    Args:
        data (np.ndarray): Data to smooth
        sigma (float): Standard deviation for Gaussian kernel
        
    Returns:
        np.ndarray: Smoothed data
    """
    return gaussian_filter1d(data, sigma=sigma)

def visualize_vector_losses(model_name, smoothing_sigma=1000000):
    """
    Visualize vector losses in a grid pattern, with 5 plots per row.
    
    Args:
        model_name (str): Name of the model
        smoothing_sigma (float): Sigma parameter for Gaussian smoothing
    """
    # Set publication-quality style
    plt.style.use('seaborn-v0_8-paper')
    mpl.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'font.size': 14,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 22
    })
    
    # Create directories if they don't exist
    os.makedirs('results/figures', exist_ok=True)
    
    # Load losses
    model_id = model_name.split('/')[-1].lower()

    losses_pattern = f'losses_{model_id}_idx*.pt'

    # Prefer the newer sub-directory first
    losses_dir = Path('results/vars/losses') if Path('results/vars/losses').exists() else Path('results/vars')
    loss_files = list(losses_dir.glob(losses_pattern))
    
    # Sort numerically by index rather than lexicographically
    loss_files = sorted(loss_files, key=lambda x: int(re.search(r'idx(\d+)\.pt$', str(x)).group(1)))
    
    if not loss_files:
        print(f"No loss files found matching pattern: {losses_pattern} in {losses_dir}")
        return
    
    vector_norms = {}

    vectors_dir = Path('results/vars/optimized_vectors')
    if vectors_dir.exists():
        vector_files = list(vectors_dir.glob(f'{model_id}_idx*.pt'))
        for vf in vector_files:
            try:
                vec_idx_match = re.search(rf'{re.escape(model_id)}_idx(\d+)\.pt$', vf.name)
                if not vec_idx_match:
                    continue
                vec_idx = int(vec_idx_match.group(1))
                vec_obj = torch.load(vf)

                # Handle different storage formats
                if torch.is_tensor(vec_obj):
                    vec_tensor = vec_obj
                elif isinstance(vec_obj, dict):
                    # If dict contains exactly one tensor value (e.g. {category: tensor})
                    tensor_values = [v for v in vec_obj.values() if torch.is_tensor(v)]
                    vec_tensor = tensor_values[0] if tensor_values else None
                else:
                    vec_tensor = None

                if vec_tensor is not None:
                    vector_norms[vec_idx] = torch.norm(vec_tensor).item()
            except Exception as e:
                print(f"Warning: could not load vector norm from {vf}: {e}")
    else:
        try:
            vectors_file = Path(f'results/vars/optimized_vectors_{model_id}.pt')
            if not vectors_file.exists():
                vectors_file = Path(f'results/vars/mean_vectors_{model_id}.pt')

            if vectors_file.exists():
                vectors_data = torch.load(vectors_file)
                if isinstance(vectors_data, dict) and 'vectors' in vectors_data:
                    vectors = vectors_data['vectors']
                elif isinstance(vectors_data, list) or isinstance(vectors_data, torch.Tensor):
                    vectors = vectors_data
                else:
                    vectors = None

                if vectors is not None:
                    for i, vec in enumerate(vectors):
                        vector_norms[i] = torch.norm(vec).item()
        except Exception as e:
            print(f"Error loading aggregated vector file: {e}")
    
    # Calculate grid dimensions
    n_plots = len(loss_files)
    n_cols = 7  # 5 plots per row (already set)
    n_rows = math.ceil(n_plots / n_cols)
    
    # Create a figure with proper gridspec for centering the last row
    fig = plt.figure(figsize=(5*n_cols, 4*n_rows), dpi=300, facecolor='white')
    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig, hspace=0.3, wspace=0.25)
    
    # Calculate plots in last row and offset for centering
    plots_in_last_row = n_plots - (n_rows - 1) * n_cols
    offset = (n_cols - plots_in_last_row) // 2 if plots_in_last_row > 0 and plots_in_last_row < n_cols else 0
    
    # Create axes with the grid spec
    axes = []
    for row in range(n_rows):
        for col in range(n_cols):
            # For the last row, apply the offset to center the plots
            if row == n_rows - 1 and plots_in_last_row < n_cols:
                if col >= offset and col < offset + plots_in_last_row:
                    ax = fig.add_subplot(gs[row, col])
                    axes.append(ax)
                else:
                    # Add a placeholder for empty spots
                    ax = fig.add_subplot(gs[row, col])
                    ax.set_visible(False)
                    axes.append(ax)
            else:
                ax = fig.add_subplot(gs[row, col])
                axes.append(ax)
    
    # Use a more professional colormap
    cmap = plt.cm.viridis
    
    # Process each loss file
    for idx, loss_file in enumerate(loss_files):
        if idx >= n_plots:
            break
            
        # Load losses
        losses = torch.load(loss_file)
        
        # Get the vector index from the filename
        vec_idx = int(re.search(r'idx(\d+)\.pt$', str(loss_file)).group(1))
        
        # Calculate the correct axis index for this plot
        if idx >= (n_rows - 1) * n_cols and plots_in_last_row < n_cols:
            # This is in the last row - need to adjust for centering
            row = n_rows - 1
            relative_col = idx - (n_rows - 1) * n_cols
            col = relative_col + offset
            ax_idx = row * n_cols + col
        else:
            ax_idx = idx
        
        # Get current axis
        ax = axes[ax_idx]
        ax.set_facecolor('white')
        
        # Get color from colormap
        color = cmap(idx / n_plots)
                
        # Find best learning rate based on final evaluation loss
        best_lr = list(losses.keys())[0]
        best_losses = losses[best_lr][0]
        
        # Process training losses
        train_losses = np.array(best_losses['train_losses'])
        
        batch_smoothing_sigma = float(smoothing_sigma)
        smoothed_train = smooth_data(train_losses, sigma=batch_smoothing_sigma)
        
        # Plot training losses with smoothing
        ax.plot(smoothed_train, 
                color=color,
                linewidth=2.5,
                linestyle='-',
                label='Training')
        
        # Process evaluation losses if available
        if 'eval_losses' in best_losses and best_losses['eval_losses']:
            eval_losses = np.array(best_losses['eval_losses'])
            
            # For evaluation losses, we might have fewer points (computed every 5 batches)
            # So we need to interpolate to match the training loss length
            if len(eval_losses) > 0:
                # Create x-coordinates for evaluation losses
                eval_x = np.linspace(0, len(train_losses) - 1, len(eval_losses))
                train_x = np.arange(len(train_losses))
                
                # Interpolate evaluation losses to match training loss length
                from scipy.interpolate import interp1d
                if len(eval_losses) > 1:
                    eval_interpolator = interp1d(eval_x, eval_losses, kind='linear', 
                                               bounds_error=False, fill_value='extrapolate')
                    eval_losses_interpolated = eval_interpolator(train_x)
                else:
                    # If only one eval loss, repeat it
                    eval_losses_interpolated = np.full_like(train_losses, eval_losses[0])
                
                # Calculate smoothed evaluation losses
                smoothed_eval = smooth_data(eval_losses_interpolated, sigma=batch_smoothing_sigma)
                
                # Plot evaluation losses with smoothing
                ax.plot(smoothed_eval,
                        color=color,
                        linewidth=2.5,
                        linestyle='--',
                        alpha=0.8,
                        label='Evaluation')
        else:
            # If no evaluation losses, just plot training
            pass
                
        # Set title and labels with norm and best lr in parentheses if available
        title_parts = [f'Vector {vec_idx}']
        if vec_idx in vector_norms:
            title_parts.append(f'norm: {vector_norms[vec_idx]:.3f}')
        title_parts.append(f'best lr: {best_lr:.2e}')
        
        ax.set_title(' | '.join(title_parts), fontweight='bold', pad=10)
                    
        # Set y-label for leftmost subplots
        if ax_idx % n_cols == 0:
            ax.set_ylabel('Loss', labelpad=8)
        
        # Set x-label for bottom subplots
        if ax_idx >= (n_rows - 1) * n_cols:
            ax.set_xlabel('Batch', labelpad=8)
        
        # Remove offset on x-axis
        ax.margins(x=0)
        
        # Add box and grid with cleaner visibility
        for spine in ax.spines.values():
            spine.set_linewidth(1.0)
            spine.set_color('black')
        
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
        
        # Enhanced grid settings for cleaner look
        ax.grid(True, 
                linestyle='-',
                alpha=0.15,
                color='black',
                which='major')
        
        # Add legend for first plot only
        if idx == 0:
            ax.legend(frameon=True, fancybox=True, framealpha=0.9, loc='upper right')
    
    # Add a common title for all subplots
    model_name_clean = model_id.replace("-", " ").title()
    fig.suptitle(f'Vector Training Losses - {model_name_clean}', 
                 fontweight='bold', 
                 y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save figure with high quality
    output_path = f'results/figures/vector_losses_{model_id}.pdf'
    plt.savefig(output_path,
                dpi=400,
                bbox_inches='tight',
                facecolor='white',
                edgecolor='none')
    
    print(f"Figures saved to {output_path} and {output_path.replace('.pdf', '.png')}")
    plt.show()
    plt.close()

if __name__ == "__main__":
    
    visualize_vector_losses(args.model, 
                           args.smoothing_sigma) 

# %%
