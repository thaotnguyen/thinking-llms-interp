from openai import OpenAI
import dotenv
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from nnsight import NNsight
from tqdm import tqdm
import gc

dotenv.load_dotenv(".env")

def chat(prompt, image=None):
    client = OpenAI(
        organization="org-E6iEJQGSfb0SNHMw6NFT1Cmi",
    )
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }
        ],
        temperature=0.01,
    )
    return response.choices[0].message.content

def load_model_and_vectors(compute_features=True, model_name="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"):
    """
    Load model, tokenizer and mean vectors. Optionally compute feature vectors.
    
    Args:
        compute_features (bool): If True, compute and return feature vectors by subtracting overall mean
        model_name (str): Name/path of the model to load
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    model = NNsight(model).to("cuda")
    
    # Get model identifier for file naming
    model_id = model_name.split('/')[-1].lower()
    mean_vectors_dict = torch.load(f"data/mean_vectors_{model_id}.pt")
    
    if compute_features:
        # Compute feature vectors by subtracting overall mean
        overall_mean = mean_vectors_dict['overall']['mean']
        feature_vectors = {}
        
        for label in mean_vectors_dict:
            if label != 'overall':
                feature_vectors[label] = mean_vectors_dict[label]['mean'] - overall_mean
        
        return model, tokenizer, feature_vectors
    
    return model, tokenizer, mean_vectors_dict

def custom_generate_with_projection_removal(model, tokenizer, input_ids, max_new_tokens, label, feature_vectors, layers=[10], coefficient=0.1, steer_positive=False, show_progress=True):
    """
    Generate text while removing or adding projections of specific features.
    
    Args:
        model: The model to use for generation
        tokenizer: The tokenizer
        input_ids: Input token ids
        max_new_tokens: Maximum number of tokens to generate
        label: The label to steer towards/away from
        feature_vectors: Dictionary of feature vectors
        steer_positive: If True, steer towards the label, if False steer away
        show_progress: If True, show progress bar
    """
    generated_ids = input_ids.clone().cpu()
    if label in feature_vectors:
        feature_vector = feature_vectors[label].to("cuda").to(torch.bfloat16)
    else:
        feature_vector = None
    
    iterator = range(max_new_tokens)
    if show_progress:
        iterator = tqdm(iterator, desc="Generating response")
            
    for k in iterator:
        input_chunk = generated_ids.to("cuda")
        
        with torch.no_grad():
            with model.trace(input_chunk) as trace:
                # First run the model normally to get hidden states
                outputs = model.lm_head.output.save()
                
                if feature_vector is not None:
                    for layer_idx in layers:
                        
                        if steer_positive:
                            expanded_feature = feature_vector[layer_idx].unsqueeze(0).unsqueeze(0).expand(1, input_chunk.size(1)-1, -1)
                            model.model.layers[layer_idx].output[0][:, 1:] += coefficient * expanded_feature
                        else:
                            expanded_feature = feature_vector[layer_idx].unsqueeze(0).unsqueeze(0).expand(1, input_chunk.size(1)-1, -1)
                            model.model.layers[layer_idx].output[0][:, 1:] -= coefficient * expanded_feature
        
        next_token = outputs[:, -1, :].argmax(dim=-1)
        
        if next_token.item() == tokenizer.eos_token_id:
            break
        
        generated_ids = torch.cat([generated_ids, next_token.unsqueeze(0).cpu()], dim=1)

        del trace, outputs, next_token, input_chunk
       
        torch.cuda.empty_cache()
        if k % 50 == 0:
            gc.collect()
    
    gc.collect()
    return generated_ids.cpu()