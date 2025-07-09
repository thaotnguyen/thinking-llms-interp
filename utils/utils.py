import dotenv
dotenv.load_dotenv("../.env")

import torch
from nnsight import LanguageModel
import time
import anthropic
from openai import OpenAI
import json
import re
import numpy as np
import sys
import os
import random
import pickle
from tqdm import tqdm
from chat_limiter import (
    ChatLimiter,
    process_chat_completion_batch,
    create_chat_completion_requests,
    BatchConfig
)

def print_and_flush(message):
    """Prints a message and flushes stdout."""
    print(message)
    sys.stdout.flush()

def chat(prompt, model="gpt-4.1", max_tokens=28000):

    model_provider = ""

    if model in ["gpt-4o", "gpt-4.1"]:
        model_provider = "openai"
        client = OpenAI()
    elif model in ["claude-3-opus", "claude-3-7-sonnet", "claude-3-5-haiku"]:
        model_provider = "anthropic"
        client = anthropic.Anthropic()
    elif model in ["deepseek-v3", "gemini-2-0-think", "gemini-2-0-flash", "deepseek-r1"]:
        model_provider = "openrouter"
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )

    # try 3 times with 3 second sleep between attempts
    for _ in range(3):
        try:
            if model_provider == "openai":
                client = OpenAI()
                response = client.chat.completions.create(
                    model=model,
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
                    max_completion_tokens=max_tokens,
                    temperature=1e-19,
                )
                return response.choices[0].message.content
            elif model_provider == "anthropic":
                model_mapping = {
                    "claude-3-opus": "claude-3-opus-latest",
                    "claude-3-7-sonnet": "claude-3-7-sonnet-latest",
                    "claude-3-5-haiku": "claude-3-5-haiku-latest"
                }

                if model == "claude-3-7-sonnet":
                    response = client.messages.create(
                        model=model_mapping[model],
                        temperature=1,
                        messages=[
                            {
                                "role": "user", 
                                "content": [
                                    {
                                        "type": "text",
                                        "text": prompt
                                    }
                                ]
                            }
                        ],
                        thinking = {
                            "type": "enabled",
                            "budget_tokens": max_tokens
                        },
                        max_tokens=max_tokens+1
                    )

                    thinking_response = response.content[0].thinking
                    answer_response = response.content[1].text

                    return f"<think>{thinking_response}\n</think>\n{answer_response}"

                else:
                    response = client.messages.create(
                        model=model_mapping[model],
                        temperature=1e-19,
                        messages=[
                            {
                                "role": "user", 
                                "content": [
                                    {
                                        "type": "text",
                                        "text": prompt
                                    }
                                ]
                            }
                        ],
                        max_tokens=max_tokens
                    )

                    return response.content[0].text
            elif model_provider == "openrouter":
                # Map model names to OpenRouter model IDs
                model_mapping = {
                    "deepseek-r1": "deepseek/deepseek-r1",
                    "deepseek-v3": "deepseek/deepseek-chat",
                    "gemini-2-0-think": "google/gemini-2.0-flash-thinking-exp:free",
                    "gemini-2-0-flash": "google/gemini-2.0-flash-001"
                }
                
                response = client.chat.completions.create(
                    model=model_mapping[model],
                    extra_body={},
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=1e-19,
                    max_tokens=max_tokens
                )

                if hasattr(response.choices[0].message, "reasoning"):
                    thinking_response = response.choices[0].message.reasoning
                    answer_response = response.choices[0].message.content

                    return f"<think>{thinking_response}\n</think>\n{answer_response}"
                else:
                    return response.choices[0].message.content
            
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(20)

    return None

async def chat_batch(prompts, model="gpt-4.1", max_tokens=28000, max_concurrent_requests=30, max_retries_per_item=3):
    """
    Process a batch of prompts using the chat_limiter library for parallel processing.
    
    Args:
        prompts (list): List of prompts to process
        model (str): Model to use for the chat
        max_tokens (int): Maximum number of tokens per response
        max_concurrent_requests (int): Maximum number of concurrent requests
        max_retries_per_item (int): Maximum number of retries per item
        
    Returns:
        list: List of responses corresponding to the prompts
    """
    # Create chat completion requests
    requests = create_chat_completion_requests(
        model=model,
        prompts=prompts,
        max_tokens=max_tokens,
        temperature=1e-19,
    )
    
    # Configure batch processing
    config = BatchConfig(
        max_concurrent_requests=max_concurrent_requests,
        max_retries_per_item=max_retries_per_item,
        group_by_model=True,
        verbose=True
    )
    
    # Process batch with increased timeout for reliability
    async with ChatLimiter.for_model(model, timeout=180.0) as limiter:
        results = await process_chat_completion_batch(limiter, requests, config)
    
    # Extract responses and handle errors
    responses = []
    for i, result in enumerate(results):
        if result.success:
            # Handle different response formats based on model
            response = result.result
            if hasattr(response, 'choices') and response.choices:
                content = response.choices[0].message.content
                # Handle thinking models that might have reasoning
                if hasattr(response.choices[0].message, 'reasoning') and response.choices[0].message.reasoning:
                    thinking_response = response.choices[0].message.reasoning
                    responses.append(f"<think>{thinking_response}\n</think>\n{content}")
                else:
                    responses.append(content)
            else:
                responses.append(str(response))
        else:
            # Fallback to individual chat call for failed requests
            print(f"Batch request {i} failed: {result.error}. Falling back to individual chat call.")
            fallback_response = chat(prompts[i], model=model, max_tokens=max_tokens)
            responses.append(fallback_response)
    
    return responses

def get_char_to_token_map(text, tokenizer):
    """Create a mapping from character positions to token positions"""
    token_offsets = tokenizer.encode_plus(text, return_offsets_mapping=True)['offset_mapping']
    
    # Create mapping from character position to token index
    char_to_token = {}
    for token_idx, (start, end) in enumerate(token_offsets):
        for char_pos in range(start, end):
            char_to_token[char_pos] = token_idx
            
    return char_to_token

def process_saved_responses(model_name, n_examples, model, tokenizer, layer):
    """Load and process saved responses to get activations"""
    
    # Create filename for caching based on parameters
    model_id = model_name.split('/')[-1].lower()
    pickle_filename = f"../generate-responses/results/vars/activations_{model_id}_{n_examples}_{layer}.pkl"
    
    # Check if cached file exists
    if os.path.exists(pickle_filename):
        print(f"Loading cached activations from {pickle_filename}...")
        with open(pickle_filename, 'rb') as f:
            return pickle.load(f)
    
    print(f"Processing saved responses for {model_name}...")
    
    # Load model and tokenizer
    responses_json_path = f"../generate-responses/results/vars/responses_{model_id}.json"
    
    print(f"Loading responses from {responses_json_path}...")
    with open(responses_json_path, 'r') as f:
        responses_data = json.load(f)
    
    # Limit to n_examples
    random.shuffle(responses_data)
    responses_data = responses_data[:n_examples]
        
    # Extract text segments and their activations
    all_activations = []
    all_texts = []
    
    overall_running_mean = torch.zeros(1, model.config.hidden_size)
    overall_running_count = 0

    print(f"Extracting activations for {n_examples} sentences...")
    for response_data in tqdm(responses_data):
        if not response_data.get("thinking_process"):
            continue
            
        # Get the thinking process text
        thinking_text = response_data["thinking_process"]
        full_response = response_data["full_response"]
        
        # Split into sentences using regex
        sentences = split_into_sentences(thinking_text)
        
        # Encode the full response to get input_ids
        input_ids = tokenizer.encode(full_response, return_tensors="pt").to(model.device)
        
        # Get layer activations
        with model.trace({
            "input_ids": input_ids, 
            "attention_mask": (input_ids != tokenizer.pad_token_id).long()
        }) as tracer:
            layer_outputs = model.model.layers[layer].output[0].save()
        
        # Convert layer outputs to numpy arrays
        layer_outputs = layer_outputs.detach().to(torch.float32)
        
        # Create character to token mapping
        char_to_token = get_char_to_token_map(full_response, tokenizer)
        
        # Process each sentence
        min_token_start = float('inf')
        max_token_end = -float('inf')
        for sentence in sentences:
            # Find this sentence in the original text
            text_pos = full_response.find(sentence)
            if text_pos >= 0:
                # Get start and end token positions
                token_start = char_to_token.get(text_pos, None)
                token_end = char_to_token.get(text_pos + len(sentence), None)
                
                if token_start is not None and token_end is not None and token_start < token_end:
                    if token_start < min_token_start:
                        min_token_start = token_start
                    if token_end > max_token_end:
                        max_token_end = token_end

                    # Extract activations for this segment
                    segment_activations = layer_outputs[:, token_start-1:token_end, :].mean(dim=1).cpu().numpy()  # Average over tokens
                                        
                    # Save the result
                    all_activations.append(segment_activations)  # Store as numpy array
                    all_texts.append(sentence)
    
        if min_token_start < layer_outputs.shape[1] and max_token_end > 0:
            vector = layer_outputs[:,min_token_start:max_token_end,:].mean(dim=1).cpu()
            overall_running_mean = overall_running_mean + (vector - overall_running_mean) / (overall_running_count + 1)
            overall_running_count += 1

    print(f"Found {len(all_activations)} sentences with activations across {overall_running_count} examples")

    # Convert overall_running_mean to numpy as well
    overall_running_mean = overall_running_mean.cpu().numpy()
    
    # Save to pickle file
    result = (all_activations, all_texts, overall_running_mean)
    with open(pickle_filename, 'wb') as f:
        pickle.dump(result, f)
    print(f"Saved activations to {pickle_filename}")

    return all_activations, all_texts, overall_running_mean


def load_model(device="cuda:0", load_in_8bit=False, model_name="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"):
    """
    Load model, tokenizer and mean vectors. Optionally compute feature vectors.
    
    Args:
        load_in_8bit (bool): If True, load the model in 8-bit mode
        model_name (str): Name/path of the model to load
    """
    model = LanguageModel(model_name, dispatch=True, load_in_8bit=load_in_8bit, device_map=device, torch_dtype=torch.bfloat16)
    
    model.generation_config.temperature=None
    model.generation_config.top_p=None
    model.generation_config.top_k=None
    model.generation_config.do_sample=False
    
    tokenizer = model.tokenizer

    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    return model, tokenizer

def custom_generate_with_steering(model, tokenizer, input_ids, max_new_tokens, steering_vector=None, layer=None, normalize=False, coefficient=1.0):
    """
    Generate text while steering with a specific feature vector.
    
    Args:
        model: The model to use for generation
        tokenizer: The tokenizer
        input_ids: Input token ids
        max_new_tokens: Maximum number of tokens to generate
        steering_vector: Vector to use for steering (should match model hidden size)
        layer: Layer index to apply steering to
        coefficient: Strength of steering (higher values = stronger effect)
        sae: Sparse Autoencoder model to use for activation-based steering
    """
    model_layers = model.model.layers

    with model.generate(
        {
            "input_ids": input_ids, 
            "attention_mask": (input_ids != tokenizer.pad_token_id).long()
        },
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
    ) as tracer:
        # Apply .all() to model to ensure interventions work across all generations
        model_layers.all()

        if steering_vector is not None and layer is not None:
            # Convert steering vector to correct device and dtype if needed
            steering_vector = steering_vector.to(model.device).to(model.dtype)
            avg_norm = model.model.layers[layer].output[0][:, 1:, :].norm(dim=-1).mean(dim=1)
            if normalize:
                steering_vector = steering_vector.unsqueeze(0).unsqueeze(0) * avg_norm
            model.model.layers[layer].output[0][:, 1:, :] += coefficient * steering_vector
        
        outputs = model.generator.output.save()
                    
    return outputs

def get_random_distinct_colors(labels):
    """
    Generate random distinct ANSI colors for each label.
    
    Args:
        labels: List of label names
        
    Returns:
        Dictionary mapping labels to ANSI color codes
    """
    # List of distinct ANSI colors (excluding black, white, and hard-to-see colors)
    # Format is "\033[COLORm" where COLOR is a number between 31-96
    distinct_colors = [
        "\033[31m",  # Red
        "\033[32m",  # Green
        "\033[33m",  # Yellow
        "\033[34m",  # Blue
        "\033[35m",  # Magenta
        "\033[36m",  # Cyan
        "\033[91m",  # Bright Red
        "\033[92m",  # Bright Green
        "\033[93m",  # Bright Yellow
        "\033[94m",  # Bright Blue
        "\033[95m",  # Bright Magenta
        "\033[96m",  # Bright Cyan
    ]
    
    # Shuffle the colors to randomize them
    random.shuffle(distinct_colors)
    
    # Ensure we have enough colors
    if len(labels) > len(distinct_colors):
        # If we need more colors, create additional ones with random RGB values
        additional_needed = len(labels) - len(distinct_colors)
        for _ in range(additional_needed):
            # Generate random RGB foreground color (38;2;r;g;b)
            r, g, b = random.randint(50, 255), random.randint(50, 255), random.randint(50, 255)
            # Ensure colors are distinct by checking minimum distance from existing colors
            # (simplified approach)
            distinct_colors.append(f"\033[38;2;{r};{g};{b}m")
    
    # Assign colors to labels
    label_colors = {}
    for i, label in enumerate(labels):
        label_colors[label] = distinct_colors[i % len(distinct_colors)]
    
    return label_colors

# Create NumpyEncoder for JSON serialization
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)


# Function to convert numpy types to Python native types
def convert_numpy_types(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list) or isinstance(obj, tuple):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


def process_batch_annotations(thinking_processes):
    """Annotate a batch of reasoning chains using the 7-category reasoning framework."""
    annotated_responses = []
    for thinking in thinking_processes:
        annotated_response = chat(f"""
Please annotate the following reasoning trace by marking segments with categories from the reasoning framework below. Use this format: ["<category>"] ... ["<end-section>"]. A sentence can be split into multiple segments if it exhibits different behaviors. Only use the categories provided below.

Reasoning Framework Categories:

1. problem-identification-framing – Problem Identification and Framing
   *Description:* This reflects the model's initial orientation toward the problem—an explicit commitment to focus attention on a particular question or task. It's not solving yet; it's mentally staking out the terrain and clarifying the goal.
   *Includes:* Explicit declarations of the question or topic to be addressed; clarifying scope or rephrasing the goal of the reasoning.
   *Excludes:* Any move toward analysis, solution generation, or speculation.
   *Examples:* "Okay, so I'm trying to figure out how pressure affects the boiling point of water.", "Okay, so I'm trying to figure out the ripple effects of making college education free."

2. metacognitive-setup – Metacognitive Setup and Decomposition Initiation
   *Description:* This captures the model's pre-analytic cognitive preparation—noticing uncertainty or complexity and deciding to plan, organize, or scaffold the reasoning process before diving in.
   *Includes:* Metacognitive statements about strategy or planning; moves to mentally break a problem into manageable parts.
   *Excludes:* Execution of any actual reasoning steps or guesses.
   *Examples:* "Hmm, let me think about this step by step.", "Let me try to visualize this.", "I'm not entirely sure where to start, but I think it's important to break it down step by step."

3. stepwise-calculation – Stepwise Calculation / Enumeration / Local Inference
   *Description:* This cluster captures the model's mechanistic reasoning—applying rules, performing arithmetic, listing possibilities. It's executing a mental algorithm.
   *Includes:* Arithmetic, combinatorics, enumeration of cases; explicit inferences from rules or facts.
   *Excludes:* High-level summaries or contextual reasoning.
   *Examples:* "3 times 7 is 21, and 21 times 11 is 231.", "So, the probability of drawing a red on the first draw is 4 out of 7, which is 4/7.", "Each face is a base for one pyramid, so 6 pyramids."

4. generating-alternatives – Generating Alternatives / Hypotheses
   *Description:* This cluster reflects the model's attempt to expand the hypothesis space. It's not committing to an answer—it's surfacing possible explanations, mechanisms, or paths forward.
   *Includes:* Generative thinking under uncertainty; multiple speculative branches or mechanisms.
   *Excludes:* Final answers or rule-based deductions.
   *Examples:* "Or maybe it's about controlling invasive species.", "Maybe it's just the body's way of fighting off the infection.", "I should also consider different scenarios."

5. information-seeking – Information-Seeking and Epistemic Uncertainty
   *Description:* The model confronts a knowledge gap and initiates action to resolve it. This is a pivot away from internal reasoning toward acquiring more information.
   *Includes:* Statements of uncertainty paired with information-seeking intent; declarations that external info is needed.
   *Excludes:* Internal speculation without intent to learn more; passive confusion without action.
   *Examples:* "I should probably look up some information to get a better understanding.", "Maybe I should ask someone or look it up to find out more information.", "I think I'll just have to check online or maybe ask a friend."

6. consequence-projection – Consequence Projection / Scenario Elaboration
   *Description:* This is forward simulation. The model is running a mental model of the world to ask: "What would happen if...?"
   *Includes:* Counterfactuals, conditionals, and policy simulation; exploration of second- or third-order effects.
   *Excludes:* Simple cause-effect or binary conclusions.
   *Examples:* "Also, with more free time, people might pursue further education.", "If the species affects farming, there might be compensation programs.", "Cities might save money on road repairs due to AVs."

7. conclusion-articulation – (Sub)-Conclusion Articulation
   *Description:* This is the "wrap up this step" reflex. It's when the model finishes part of the reasoning and states a result—before continuing onward.
   *Includes:* (Partial) conclusions or intermediate inferences; logic checkpoints or sanity checks.
   *Excludes:* Problem framing or speculative reasoning.
   *Examples:* "So, each face is a base for one pyramid, so 6 pyramids.", "So, the next month is December, which is D.", "So, if the surgeon is the mother, then yes, the patient is her son."

Reasoning trace to annotate:
{thinking}

Only return the annotated text using the specified format. Do not include any explanation or commentary outside the annotations.
If the last sentence is not finished, do not include it in the annotations.
""")
        annotated_responses.append(annotated_response)
    
    return annotated_responses


model_mapping = {
    "meta-llama/Llama-3.1-8B":"deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "Qwen/Qwen2.5-Math-1.5B":"deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "Qwen/Qwen2.5-14B":"deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
}

def split_into_sentences(text, min_words=3):
    """
    Split text into sentences and filter based on quality criteria.
    
    Args:
        text (str): The input text to split into sentences
        
    Returns:
        list: List of cleaned sentences with at least 3 words each
    """
    # Split on sentence-ending punctuation, newlines, but avoid splitting on decimal numbers and single letter abbreviations
    # The regex matches: 
    # - (?<!\b\w)[!;] for exclamation and semicolons not preceded by single letters (avoids "k!" splits)
    # - \? for question marks (always split)
    # - (?<!\d)(?<!\b\w)\.(?!\d) matches periods not between digits and not after single letters (avoids "E." splits)
    # - (?<=\d)\.(?=\s|"|$) matches periods after digits followed by space, quote, or end (like "$5,000.")
    # - \n+ matches one or more newlines
    sentences = re.split(r'(?<!\b\w)[!;]|\?|(?<!\d)(?<!\b\w)\.(?!\d)|(?<=\d)\.(?=\s|"|$)|\n+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    sentences = [s for s in sentences if len(s.split()) >= min_words]
    
    # Post-processing: Handle sentences that start with quotes after period splits
    # If a sentence starts with a quote, move it to the end of the previous sentence
    processed_sentences = []
    for i, sentence in enumerate(sentences):
        if i > 0 and sentence.startswith('"') and processed_sentences:
            # Move the quote to the previous sentence and remove it from current
            processed_sentences[-1] += '"'
            current_sentence = sentence[1:].strip()  # Remove quote and leading space
            if current_sentence and len(current_sentence.split()) >= min_words:
                processed_sentences.append(current_sentence)
        else:
            processed_sentences.append(sentence)
    
    return processed_sentences


