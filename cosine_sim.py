# %%
import torch
import matplotlib.pyplot as plt

mean_vectors_dict = torch.load("mean_vectors.pt")
feature_vectors = {}

overall_mean = mean_vectors_dict['overall']['mean']
for label in mean_vectors_dict:
    if label != 'overall':
        feature_vectors[label] = mean_vectors_dict[label]['mean'] - overall_mean

def plot_aggregated_cosine_similarity_heatmap(feature_vectors):
    labels = list(feature_vectors.keys())
    n_labels = len(labels)
    n_layers = feature_vectors[labels[0]].shape[0]
    
    # Create aggregated similarity matrix
    aggregated_similarity = torch.zeros((n_labels, n_labels))
    for layer_idx in range(n_layers):
        for i, label_1 in enumerate(labels):
            for j, label_2 in enumerate(labels):
                similarity = torch.cosine_similarity(
                    feature_vectors[label_1][layer_idx], 
                    feature_vectors[label_2][layer_idx], 
                    dim=-1
                )
                # Rescale from [-1,1] to [0,1]
                similarity = (similarity + 1) / 2
                aggregated_similarity[i, j] += similarity / n_layers  # Average across layers
    
    # Create heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(aggregated_similarity, cmap='Blues', vmin=0, vmax=1)
    plt.colorbar(label='Average Cosine Similarity')
    
    # Add labels
    plt.xticks(range(n_labels), labels, rotation=45, ha='right')
    plt.yticks(range(n_labels), labels)
    
    # Add text annotations with values
    for i in range(n_labels):
        for j in range(n_labels):
            plt.text(j, i, f'{aggregated_similarity[i, j]:.2f}',
                    ha='center', va='center',
                    color='black' if aggregated_similarity[i, j] < 0.8 else 'white')
    
    plt.title('Average Cosine Similarity Between Feature Vectors')
    plt.tight_layout()
    plt.savefig('figures/cosine_similarity_heatmap.png', dpi=300)
    plt.show()


# Plot the aggregated heatmap
plot_aggregated_cosine_similarity_heatmap(feature_vectors)
# %%
