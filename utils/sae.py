import torch.nn as nn
import os
import torch

def load_sae(model_id, layer, n_clusters, load_base_decoder=False):
    sae_path = f'../train-saes/results/vars/saes/sae_{model_id}_layer{layer}_clusters{n_clusters}.pt'
    if not os.path.exists(sae_path):
        raise FileNotFoundError(f"SAE model not found at {sae_path}")
        
    checkpoint = torch.load(sae_path)
    
    # Create SAE model
    sae = SAE(checkpoint['input_dim'], checkpoint['num_latents'], k=checkpoint.get('topk', 3))
    
    # Load weights
    sae.encoder.weight.data = checkpoint['encoder_weight']
    sae.encoder.bias.data = checkpoint['encoder_bias']
    sae.W_dec.data = checkpoint['decoder_weight']
    sae.b_dec.data = checkpoint['b_dec']
    
    print(f"Loaded SAE model from {sae_path}")

    return sae, checkpoint

class SAE(nn.Module):
    def __init__(self, d_in, num_latents, k=1):
        super().__init__()
        self.encoder = nn.Linear(d_in, num_latents, bias=True)
        self.encoder.bias.data.zero_()
        self.W_dec = nn.Parameter(self.encoder.weight.data.clone())
        self.b_dec = nn.Parameter(torch.zeros(d_in))
        self.k = k
        self.set_decoder_norm_to_unit_norm()
        
    @torch.no_grad()
    def set_decoder_norm_to_unit_norm(self):
        norm = torch.norm(self.W_dec.data, dim=1, keepdim=True)
        self.W_dec.data /= norm + 1e-5
        
    def encode(self, x):
        forward = self.encoder(x - self.b_dec)
        top_acts, top_indices = forward.topk(self.k, dim=-1)
        return top_acts, top_indices
        
    def decode(self, top_acts, top_indices):
        batch_size = top_indices.shape[0]
        
        # Reshape for embedding_bag
        top_acts_flat = top_acts.view(-1)
        top_indices_flat = top_indices.view(-1)
        
        # For embedding_bag we need offsets that point to the start of each sample
        offsets = torch.arange(0, batch_size, device=top_indices.device) * self.k
        
        # Use embedding_bag
        res = nn.functional.embedding_bag(
            top_indices_flat, self.W_dec, offsets=offsets, 
            per_sample_weights=top_acts_flat, mode="sum"
        )
        
        return res + self.b_dec
        
    def forward(self, x):
        top_acts, top_indices = self.encode(x)
        return self.decode(top_acts, top_indices)