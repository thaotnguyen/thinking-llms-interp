# %%

%load_ext autoreload
%autoreload 2

# %%
import random
import numpy as np
import json
import torch
import gc
import matplotlib.pyplot as plt

# Import utility functions
from utils.clustering import (
    convert_numpy_types, SUPPORTED_CLUSTERING_METHODS,
    load_trained_clustering_data, predict_clusters, evaluate_clustering_scoring_metrics,
    generate_category_descriptions, evaluate_clustering_accuracy, compute_centroid_orthogonality,
    generate_representative_examples
)
from utils import utils

# Note: If you get asyncio errors in Jupyter, install nest_asyncio:
# !pip install nest_asyncio

print("Available clustering methods:", list(SUPPORTED_CLUSTERING_METHODS))

# === 4 clusters ===
# Accuracy: 0.60625
# Orthogonality: 0.8395199179649353
# === 5 clusters ===
# Accuracy: 0.657
# Orthogonality: 0.8328043222427368
# === 6 clusters ===
# Accuracy: 0.685
# Orthogonality: 0.8527795076370239
# === 7 clusters ===
# Accuracy: 0.6914285714285714
# Orthogonality: 0.8205415606498718
# === 8 clusters ===
# Accuracy: 0.73125
# Orthogonality: 0.8324916958808899
# === 9 clusters ===
# Accuracy: 0.69
# Orthogonality: 0.8549018502235413
# === 10 clusters ===
# Accuracy: 0.7095
# Orthogonality: 0.8570114970207214
# === 11 clusters ===
# Accuracy: 0.7159090909090908
# Orthogonality: 0.8306844830513
# === 12 clusters ===
# Accuracy: 0.7045833333333333
# Orthogonality: 0.8492978811264038
# === 13 clusters ===
# Accuracy: 0.7134615384615384
# Orthogonality: 0.8547097444534302
# === 14 clusters ===
# Accuracy: 0.7599999999999999
# Orthogonality: 0.8572133779525757
# === 15 clusters ===
# Accuracy: 0.6813333333333333
# Orthogonality: 0.8377721309661865
# === 16 clusters ===
# Accuracy: 0.75
# Orthogonality: 0.8672488927841187
# === 17 clusters ===
# Accuracy: 0.7605882352941177
# Orthogonality: 0.8508486151695251
# === 18 clusters ===
# Accuracy: 0.7522222222222221
# Orthogonality: 0.8360217809677124
# === 19 clusters ===
# Accuracy: 0.7232193802262664
# Orthogonality: 0.8622407913208008
# === 20 clusters ===
# Accuracy: 0.76725
# Orthogonality: 0.8693614602088928
# === 21 clusters ===
# Accuracy: 0.7492857142857143
# Orthogonality: 0.8758158087730408
# === 22 clusters ===
# Accuracy: 0.7788095238095236
# Orthogonality: 0.8752132058143616
# === 23 clusters ===
# Accuracy: 0.7438095238095238
# Orthogonality: 0.879460871219635
# === 24 clusters ===
# Accuracy: 0.7490909090909091
# Orthogonality: 0.8796516060829163
# === 25 clusters ===
# Accuracy: 0.7348
# Orthogonality: 0.8770415186882019
# === 26 clusters ===
# Accuracy: 0.7570000000000001
# Orthogonality: 0.8776075839996338
# === 27 clusters ===
# Accuracy: 0.7625925925925925
# Orthogonality: 0.8730857372283936
# === 28 clusters ===
# Accuracy: 0.7375925925925927
# Orthogonality: 0.8864976167678833
# === 29 clusters ===
# Accuracy: 0.7748214285714287
# Orthogonality: 0.8847964406013489
# === 30 clusters ===
# Accuracy: 0.7698333333333331
# Orthogonality: 0.8785412907600403
# === 31 clusters ===
# Accuracy: 0.7624999999999998
# Orthogonality: 0.8744891881942749
# === 32 clusters ===
# Accuracy: 0.78109375
# Orthogonality: 0.8779217004776001
# === 33 clusters ===
# Accuracy: 0.7789393939393942
# Orthogonality: 0.8722381591796875
# === 34 clusters ===
# Accuracy: 0.7627941176470588
# Orthogonality: 0.8795950412750244
# === 35 clusters ===
# Accuracy: 0.7821428571428575
# Orthogonality: 0.8795278668403625
# === 36 clusters ===
# Accuracy: 0.7708333333333335
# Orthogonality: 0.8796628713607788
# === 37 clusters ===
# Accuracy: 0.7708108108108107
# Orthogonality: 0.8773086071014404
# === 38 clusters ===
# Accuracy: 0.7574324324324323
# Orthogonality: 0.8868055939674377
# === 39 clusters ===
# Accuracy: 0.7830769230769231
# Orthogonality: 0.8844630718231201
# === 40 clusters ===
# Accuracy: 0.769102564102564
# Orthogonality: 0.8811087012290955
# === 41 clusters ===
# Accuracy: 0.7884146341463415
# Orthogonality: 0.8808088898658752
# === 42 clusters ===
# Accuracy: 0.7772619047619048
# Orthogonality: 0.8795025944709778
# === 43 clusters ===
# Accuracy: 0.7872619047619048
# Orthogonality: 0.8848721981048584
# === 44 clusters ===
# Accuracy: 0.7619318181818181
# Orthogonality: 0.88416588306427
# === 45 clusters ===
# Accuracy: 0.8031395348837208
# Orthogonality: 0.8892515301704407
# === 46 clusters ===
# Accuracy: 0.7846739130434779
# Orthogonality: 0.8800004124641418
# === 47 clusters ===
# Accuracy: 0.7696739130434783
# Orthogonality: 0.8880552053451538
# === 48 clusters ===
# Accuracy: 0.7868749999999998
# Orthogonality: 0.8852201104164124
# === 49 clusters ===
# Accuracy: 0.7748979591836735
# Orthogonality: 0.8842777609825134
# === 50 clusters ===
# Accuracy: 0.7662
# Orthogonality: 0.881536602973938
# === 51 clusters ===
# Accuracy: 0.8147959183673469
# Orthogonality: 0.8860558271408081
# === 52 clusters ===
# Accuracy: 0.7503
# Orthogonality: 0.8885105848312378
# === 53 clusters ===
# Accuracy: 0.756442307692308
# Orthogonality: 0.8922771215438843
# === 54 clusters ===
# Accuracy: 0.7929629629629632
# Orthogonality: 0.8830020427703857
# === 55 clusters ===
# Accuracy: 0.7840909090909091
# Orthogonality: 0.8884214758872986
# === 56 clusters ===
# Accuracy: 0.7847272727272725
# Orthogonality: 0.8954654335975647
# === 57 clusters ===
# Accuracy: 0.7877678571428571
# Orthogonality: 0.8929138779640198
# === 58 clusters ===
# Accuracy: 0.7671551724137928
# Orthogonality: 0.8897291421890259
# === 59 clusters ===
# Accuracy: 0.7858620689655176
# Orthogonality: 0.8912578821182251
# === 60 clusters ===
# Accuracy: 0.7970833333333334
# Orthogonality: 0.8883944153785706
# === 61 clusters ===
# Accuracy: 0.7771666666666666
# Orthogonality: 0.891502320766449
# === 62 clusters ===
# Accuracy: 0.7748387096774192
# Orthogonality: 0.8896616101264954
# === 63 clusters ===
# Accuracy: 0.7507142857142857
# Orthogonality: 0.8870825171470642
# === 64 clusters ===
# Accuracy: 0.7865624999999999
# Orthogonality: 0.8892046213150024

# %%
# Configuration - modify these parameters as needed
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
LAYER = 6
N_EXAMPLES = 100000
N_CLUSTERS = 22 # 29
CLUSTERING_METHOD = "sae_topk"  # Choose from SUPPORTED_CLUSTERING_METHODS
LOAD_IN_8BIT = False
N_AUTOGRADER_EXAMPLES = 100
N_DESCRIPTION_EXAMPLES = 50

# Get model identifier for file naming
model_id = MODEL_NAME.split('/')[-1].lower()

print(f"Configuration:")
print(f"Model: {MODEL_NAME}")
print(f"Model ID: {model_id}")
print(f"Layer: {LAYER}")
print(f"Number of examples: {N_EXAMPLES}")
print(f"Number of clusters: {N_CLUSTERS}")
print(f"Clustering method: {CLUSTERING_METHOD}")

# %%
# Load model and process activations
print("Loading model and processing activations...")
model, tokenizer = utils.load_model(
    model_name=MODEL_NAME,
    load_in_8bit=LOAD_IN_8BIT
)

print(f"Model loaded successfully. Model type: {type(model)}")

# %%
# Process saved responses to get activations
print("Processing saved responses...")
all_activations, all_texts, overall_mean = utils.process_saved_responses(
    MODEL_NAME, 
    N_EXAMPLES,
    model,
    tokenizer,
    LAYER
)

print(f"Processed {len(all_activations)} activations")
print(f"Activation shape: {all_activations[0].shape}")
print(f"Number of texts: {len(all_texts)}")
print(f"Overall mean shape: {overall_mean.shape}")

# Clean up GPU memory
del model, tokenizer
torch.cuda.empty_cache()
gc.collect()

# %%

# Show examples of random texts
sample_size = 500
sample = random.sample(all_texts, sample_size)
for i in range(sample_size):
    print(sample[i])
    print("-"*100)

# %%
# Center and normalize activations
print("Centering and normalizing activations...")
all_activations = [x - overall_mean for x in all_activations]
all_activations = np.stack([a.reshape(-1) for a in all_activations])
norms = np.linalg.norm(all_activations, axis=1, keepdims=True)
all_activations = all_activations / norms

print(f"Final activation matrix shape: {all_activations.shape}")
print(f"Activation norms (should be ~1.0): min={np.min(norms):.4f}, max={np.max(norms):.4f}, mean={np.mean(norms):.4f}")

# %%
# Load trained clustering data
print(f"Loading trained clustering data for {CLUSTERING_METHOD}...")
try:
    clustering_data = load_trained_clustering_data(model_id, LAYER, N_CLUSTERS, CLUSTERING_METHOD)
    print("Clustering data loaded successfully!")
    print(f"Available keys in clustering data: {list(clustering_data.keys())}")
    
    if 'cluster_centers' in clustering_data:
        cluster_centers = clustering_data['cluster_centers']
        print(f"Cluster centers shape: {cluster_centers.shape}")
    else:
        print("No cluster_centers found in clustering data")
        
except FileNotFoundError as e:
    print(f"Error: {e}")
    print("Make sure you have trained clustering models for this configuration first.")

# %%
# Predict cluster labels for current activations
print("Predicting cluster labels...")
cluster_labels = predict_clusters(all_activations, clustering_data)

print(f"Cluster labels shape: {cluster_labels.shape}")
print(f"Unique cluster labels: {np.unique(cluster_labels)}")

# Count examples per cluster
unique_labels, counts = np.unique(cluster_labels, return_counts=True)
cluster_counts = dict(zip(unique_labels, counts))
print(f"Cluster distribution: {cluster_counts}")

# %%
# Visualize cluster distribution
plt.figure(figsize=(10, 6))
plt.bar(list(cluster_counts.keys()), list(cluster_counts.values()))
plt.title(f'Cluster Distribution - {CLUSTERING_METHOD} (Layer {LAYER})')
plt.xlabel('Cluster ID')
plt.ylabel('Number of Examples')
plt.xticks(list(cluster_counts.keys()))
for i, count in cluster_counts.items():
    plt.text(i, count + 1, str(count), ha='center')
plt.show()

# %%
# Generate representative examples for each cluster
print("Generating representative examples...")
representative_examples = generate_representative_examples(
    cluster_centers, all_texts, cluster_labels, all_activations
)

print("Representative examples by cluster:")
for cluster_idx in sorted(representative_examples.keys()):
    examples = representative_examples[cluster_idx]
    print(f"\nCluster {cluster_idx} ({len(examples)} examples):")
    for i, example in enumerate(examples[:5]):  # Show top 5
        print(f"  {i+1}. {example}")
    if len(examples) > 5:
        print(f"  ... and {len(examples) - 5} more")

# %%
# Generate category descriptions
print("Generating category descriptions...")
categories = generate_category_descriptions(
    cluster_centers, MODEL_NAME, N_DESCRIPTION_EXAMPLES, representative_examples
)

print("Generated category descriptions:")
for cluster_id, title, description in categories:
    print(f"\nCluster {cluster_id}:")
    print(f"  Title: {title}")
    print(f"  Description: {description}")

# %%
# Compute centroid orthogonality
print("Computing centroid orthogonality...")
orthogonality = compute_centroid_orthogonality(cluster_centers)
print(f"Average centroid orthogonality: {orthogonality:.4f}")

# %%
# Run accuracy evaluation (binary autograder)
print("Running accuracy evaluation...")
accuracy_results = evaluate_clustering_accuracy(
    all_texts, cluster_labels, categories, N_AUTOGRADER_EXAMPLES
)

print("Accuracy evaluation results:")
if "avg" in accuracy_results:
    avg_results = accuracy_results["avg"]
    print(f"  Average accuracy: {avg_results['accuracy']:.4f}")
    print(f"  Average precision: {avg_results['precision']:.4f}")
    print(f"  Average recall: {avg_results['recall']:.4f}")
    print(f"  Average F1: {avg_results['f1']:.4f}")

print("\nPer-cluster results:")
for cluster_id, title, description in categories:
    if cluster_id in accuracy_results:
        results = accuracy_results[cluster_id]
        print(f"  Cluster {cluster_id} ({title}):")
        print(f"    Accuracy: {results.get('accuracy', 0):.4f}")
        print(f"    Precision: {results.get('precision', 0):.4f}")
        print(f"    Recall: {results.get('recall', 0):.4f}")
        print(f"    F1: {results.get('f1', 0):.4f}")

# %%
# Display detailed classification results for each cluster
print("\n" + "="*80)
print("DETAILED CLASSIFICATION ANALYSIS")
print("="*80)
print("Shows which sentences were correctly/incorrectly classified during accuracy evaluation")

for cluster_id, title, description in categories:
    cluster_id_str = str(cluster_id)
    if cluster_id_str not in accuracy_results or "classifications" not in accuracy_results[cluster_id_str]:
        continue
        
    cluster_results = accuracy_results[cluster_id_str]
    classifications = cluster_results["classifications"]
    
    print(f"\n{'='*60}")
    print(f"CLUSTER {cluster_id}: {title}")
    print(f"{'='*60}")
    print(f"Description: {description}")
    print(f"Metrics: TP={cluster_results.get('true_positives', 0)}, "
          f"FP={cluster_results.get('false_positives', 0)}, "
          f"TN={cluster_results.get('true_negatives', 0)}, "
          f"FN={cluster_results.get('false_negatives', 0)}")
    
    # Group classifications by type
    true_positives = []   # Correctly identified as belonging to cluster
    false_negatives = []  # Incorrectly identified as NOT belonging to cluster  
    true_negatives = []   # Correctly identified as NOT belonging to cluster
    false_positives = []  # Incorrectly identified as belonging to cluster
    
    # Use ground truth information stored in enhanced classifications
    for item in classifications:
        sentence_idx = item["sentence_id"]
        prediction = item["belongs_to_category"]  # "Yes" or "No"
        ground_truth = item.get("ground_truth", "Unknown")  # "Yes" or "No"
        explanation = item.get("explanation", "")
        sentence_text = item.get("sentence_text", f"Sentence {sentence_idx}")
        
        if ground_truth == "Yes":
            # Should belong to cluster
            if prediction == "Yes":
                true_positives.append((sentence_text, explanation))
            else:
                false_negatives.append((sentence_text, explanation))
        else:
            # Should NOT belong to cluster
            if prediction == "No":
                true_negatives.append((sentence_text, explanation))
            else:
                false_positives.append((sentence_text, explanation))
    
    # Display results
    print(f"\n🟢 TRUE POSITIVES ({len(true_positives)}): Correctly identified as belonging to cluster")
    for i, (sentence_text, explanation) in enumerate(true_positives[:5]):  # Show first 5
        print(f"  {i+1}. {explanation}")
        print(f"     Text: {sentence_text}")
    if len(true_positives) > 5:
        print(f"     ... and {len(true_positives) - 5} more")
    
    print(f"\n🔴 FALSE NEGATIVES ({len(false_negatives)}): Incorrectly identified as NOT belonging to cluster")
    for i, (sentence_text, explanation) in enumerate(false_negatives[:5]):  # Show first 5
        print(f"  {i+1}. {explanation}")
        print(f"     Text: {sentence_text}")
    if len(false_negatives) > 5:
        print(f"     ... and {len(false_negatives) - 5} more")
    
    print(f"\n🟢 TRUE NEGATIVES ({len(true_negatives)}): Correctly identified as NOT belonging to cluster")
    for i, (sentence_text, explanation) in enumerate(true_negatives[:3]):  # Show first 3
        print(f"  {i+1}. {explanation}")
        print(f"     Text: {sentence_text}")
    if len(true_negatives) > 3:
        print(f"     ... and {len(true_negatives) - 3} more")
    
    print(f"\n🔴 FALSE POSITIVES ({len(false_positives)}): Incorrectly identified as belonging to cluster")
    for i, (sentence_text, explanation) in enumerate(false_positives[:3]):  # Show first 3
        print(f"  {i+1}. {explanation}")
        print(f"     Text: {sentence_text}")
    if len(false_positives) > 3:
        print(f"     ... and {len(false_positives) - 3} more")

# %%
# Run comprehensive evaluation using the main evaluation function
print("Running comprehensive evaluation...")
comprehensive_results = evaluate_clustering_scoring_metrics(
    all_texts, cluster_labels, N_CLUSTERS, all_activations, cluster_centers,
    MODEL_NAME, N_AUTOGRADER_EXAMPLES, N_DESCRIPTION_EXAMPLES
)

print("Comprehensive evaluation summary:")
print(f"  Overall accuracy: {comprehensive_results['accuracy']:.4f}")
print(f"  Assignment rate: {comprehensive_results['assigned_fraction']:.4f}")
print(f"  Orthogonality: {comprehensive_results['orthogonality']:.4f}")

# %%
# Display detailed results
print("Detailed results by cluster:")
detailed_results = comprehensive_results['detailed_results']

for cluster_id in sorted(detailed_results.keys(), key=int):
    cluster_info = detailed_results[cluster_id]
    print(f"\n{'='*60}")
    print(f"CLUSTER {cluster_id}: {cluster_info['title']}")
    print(f"{'='*60}")
    print(f"Size: {cluster_info['size']} examples")
    print(f"Precision: {cluster_info['precision']:.4f}")
    print(f"Recall: {cluster_info['recall']:.4f}")
    print(f"Accuracy: {cluster_info['accuracy']:.4f}")
    print(f"F1: {cluster_info['f1']:.4f}")
    print(f"\nDescription: {cluster_info['description']}")
    print(f"\nTop examples:")
    for i, example in enumerate(cluster_info['examples'][:10]):
        print(f"  {i+1}. {example}")

# %%
# Create performance visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Extract metrics for plotting
cluster_ids = []
precisions = []
recalls = []
f1_scores = []
sizes = []

for cluster_id in sorted(detailed_results.keys(), key=int):
    cluster_info = detailed_results[cluster_id]
    cluster_ids.append(int(cluster_id))
    precisions.append(cluster_info['precision'])
    recalls.append(cluster_info['recall'])
    f1_scores.append(cluster_info['f1'])
    sizes.append(cluster_info['size'])

# Precision by cluster
axes[0, 0].bar(cluster_ids, precisions)
axes[0, 0].set_title('Precision by Cluster')
axes[0, 0].set_xlabel('Cluster ID')
axes[0, 0].set_ylabel('Precision')
axes[0, 0].set_ylim(0, 1)

# Recall by cluster
axes[0, 1].bar(cluster_ids, recalls)
axes[0, 1].set_title('Recall by Cluster')
axes[0, 1].set_xlabel('Cluster ID')
axes[0, 1].set_ylabel('Recall')
axes[0, 1].set_ylim(0, 1)

# F1 scores by cluster
axes[1, 0].bar(cluster_ids, f1_scores)
axes[1, 0].set_title('F1 Score by Cluster')
axes[1, 0].set_xlabel('Cluster ID')
axes[1, 0].set_ylabel('F1 Score')
axes[1, 0].set_ylim(0, 1)

# Cluster sizes
axes[1, 1].bar(cluster_ids, sizes)
axes[1, 1].set_title('Cluster Sizes')
axes[1, 1].set_xlabel('Cluster ID')
axes[1, 1].set_ylabel('Number of Examples')

plt.tight_layout()
plt.show()

# %%
# Save results to JSON file (optional)
save_results = input("Save results to JSON file? (y/n): ").lower().strip() == 'y'

if save_results:
    results_filename = f'interactive_evaluation_{CLUSTERING_METHOD}_{model_id}_layer{LAYER}_clusters{N_CLUSTERS}.json'
    
    # Prepare results for saving
    save_data = {
        'configuration': {
            'model_name': MODEL_NAME,
            'model_id': model_id,
            'layer': LAYER,
            'n_examples': N_EXAMPLES,
            'n_clusters': N_CLUSTERS,
            'clustering_method': CLUSTERING_METHOD,
            'n_autograder_examples': N_AUTOGRADER_EXAMPLES,
            'n_description_examples': N_DESCRIPTION_EXAMPLES
        },
        'results': convert_numpy_types(comprehensive_results),
        'cluster_distribution': convert_numpy_types(cluster_counts)
    }
    
    with open(results_filename, 'w') as f:
        json.dump(save_data, f, indent=2)
    
    print(f"Results saved to {results_filename}")

print("\nInteractive evaluation complete!")