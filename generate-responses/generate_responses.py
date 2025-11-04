import argparse
import dotenv
dotenv.load_dotenv("../.env")

import torch
import os
from datasets import load_dataset
import random
import json
import gc
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import utils
from tqdm import tqdm

MAX_TOKENS_IN_INPUT = 5000

# Prompt used to build questions from the MedCaseReasoning dataset
PROMPT_TEMPLATE = (
    "Read the following case presentation and give the most likely diagnosis.\nFirst, provide your internal reasoning for the diagnosis within the tags <think> ... </think>.\n"
    "Then, output the final diagnosis (just the name of the disease/entity) within the tags <answer> ... </answer>.\n\n"
    "----------------------------------------\n"
    "CASE PRESENTATION\n"
    "----------------------------------------\n"
    "{case_prompt}\n\n"
    "----------------------------------------\n"
    "OUTPUT TEMPLATE\n"
    "----------------------------------------\n"
    "<think>\n"
    "...your internal reasoning for the diagnosis...\n"
    "</think>"
    "<answer>\n"
    "...the name of the disease/entity...\n"
    "</answer>"
)

# Parse arguments
parser = argparse.ArgumentParser(description="Generate responses from models without steering vectors")
parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
                    help="Model to generate responses from")
parser.add_argument("--dataset", type=str, default="tmknguyen/MedCaseReasoning-filtered",
                    help="Dataset in HuggingFace to generate responses from", choices=["zou-lab/MedCaseReasoning", "tmknguyen/MedCaseReasoning-filtered"])
parser.add_argument("--dataset_split", type=str, default="train",
                    help="Split of dataset to generate responses from")
parser.add_argument("--max_tokens", type=int, default=4096,
                    help="Maximum number of tokens to generate")
parser.add_argument("--seed", type=int, default=42,
                    help="Random seed")
parser.add_argument("--is_base_model", action="store_true", default=False,
                    help="Whether the model is a base model")
parser.add_argument("--tensor_parallel_size", type=int, default=-1,
                    help="Number of GPUs to use for tensor parallelism")
parser.add_argument("--dtype", type=str, default="auto",
                    help="Data type for model")
parser.add_argument("--temperature", type=float, default=0.0,
                    help="Temperature for sampling")
parser.add_argument("--top_p", type=float, default=1.0,
                    help="Top-p for sampling")
parser.add_argument("--engine", type=str, default="nnsight", choices=["nnsight", "vllm", "hf_endpoint"],
                    help="Generation engine to use")
parser.add_argument("--load_in_8bit", action="store_true", default=False,
                    help="Load model in 8-bit (nnsight engine only, local only)")
parser.add_argument("--remote", action="store_true", default=False,
                    help="Use NDIF remote execution (nnsight engine only)")
parser.add_argument("--api_key", type=str, default=None,
                    help="API key: NDIF key for nnsight remote, or HF token for hf_endpoint")
parser.add_argument("--provider_policy", type=str, default="auto", choices=["auto", "fastest", "cheapest"],
                    help="Provider selection policy for hf_endpoint: auto (default), fastest, or cheapest")
parser.add_argument("--batch_size", type=int, default=64,
                    help="Micro-batch size for nnsight generation only")
parser.add_argument("--max_examples", type=int, default=None,
                    help="Maximum number of examples to process (for testing)")
args, _ = parser.parse_known_args()

if args.tensor_parallel_size == -1:
    args.tensor_parallel_size = torch.cuda.device_count()

def get_prompts(tokenizer, messages_list):
    if args.is_base_model:
        prompts = [msg["content"] for msg in messages_list]
    else:
        prompts = [tokenizer.apply_chat_template([msg], tokenize=False, add_generation_prompt=True) for msg in messages_list]
    prompts_above_max_tokens = [prompt for prompt in prompts if len(prompt) > MAX_TOKENS_IN_INPUT]
    if len(prompts_above_max_tokens) > 0:
        print(f"There are {len(prompts_above_max_tokens)} prompts above MAX_TOKENS_IN_INPUT")
        for prompt in prompts_above_max_tokens:
            print(f"Length: {len(prompt)}")
        raise ValueError(f"There are {len(prompts_above_max_tokens)} prompts above MAX_TOKENS_IN_INPUT")
    return prompts

def get_messages_from_dataset(dataset_name, rows) -> dict[str, dict[str, str]]:
    """Build per-question message dicts from a HF dataset split.

    For tmknguyen/MedCaseReasoning-filtered, compose the question using PROMPT_TEMPLATE
    with the row's case_prompt, and store the gold answer from final_diagnosis.
    The question_id will be a stable composite of pmcid and row index.
    """
    if dataset_name != "tmknguyen/MedCaseReasoning-filtered":
        raise ValueError(f"Dataset {dataset_name} not supported")

    messages_by_question_id: dict[str, dict[str, str]] = {}
    for i, row in enumerate(rows):
        case_prompt = row["case_prompt"]
        question_text = PROMPT_TEMPLATE.format(case_prompt=case_prompt)

        # Prefer pmcid if available for traceability; fall back to row index
        pmcid = row.get("pmcid", None)
        question_id = f"{pmcid}_{i}" if pmcid is not None else str(i)

        messages_by_question_id[question_id] = {
            "role": "user",
            "content": question_text,
            # Lightweight category marker
            "category": "diagnosis",
            # Store the natural-language question text for downstream assertions
            "question": question_text,
            # Include gold answer for evaluation
            "gold_answer": row.get("final_diagnosis", ""),
        }
    return messages_by_question_id

def process_model_output_batch_vllm(messages_batch, tokenizer, model):
    """Get model output for a batch of messages using vLLM"""
    # Lazy import to avoid requiring vLLM when using nnsight
    from vllm import SamplingParams
    prompts = get_prompts(tokenizer, messages_batch)

    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    request_outputs = model.generate(prompts, sampling_params)
    full_responses = [request_output.prompt + request_output.outputs[0].text for request_output in request_outputs]

    # Assert the questions are in the responses
    for message, response in zip(messages_batch, full_responses):
        assert message["question"] in response, f"Question {message['question']} not in response {response}"
    
    return full_responses


def process_model_output_batch_nnsight(messages_batch, tokenizer, model, remote=False):
    """Get model output for a batch of messages using nnsight/HF generate.

    Returns full responses including the original prompt text, to mirror vLLM behavior
    and keep downstream assertions consistent.
    """
    prompts = get_prompts(tokenizer, messages_batch)

    # Tokenize with left padding as set in utils.load_model
    encoded = tokenizer(prompts, return_tensors="pt", padding=True)
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    if not remote:
        input_ids = input_ids.to(model.device)
        attention_mask = attention_mask.to(model.device)

    # Shape assertions per research guidelines
    assert input_ids.dim() == 2, f"Expected 2D input_ids, got {input_ids.shape}"
    assert attention_mask.shape == input_ids.shape, f"attention_mask shape {attention_mask.shape} must match input_ids {input_ids.shape}"

    with model.generate({
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }, max_new_tokens=args.max_tokens, pad_token_id=tokenizer.pad_token_id, temperature=args.temperature, top_p=args.top_p, remote=remote) as gen:
        outputs = model.generator.output.save()

    assert outputs.shape[0] == input_ids.shape[0], "Batch size mismatch between outputs and inputs"

    # Compute per-sample prompt lengths (number of non-pad tokens)
    prompt_lengths = attention_mask.sum(dim=1)
    full_responses = []
    for i in range(outputs.shape[0]):
        prompt_len = int(prompt_lengths[i].item())
        assert prompt_len > 0, f"Empty prompt length for sample {i}"
        new_tokens = outputs[i][prompt_len:]
        gen_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        full_responses.append(prompts[i] + gen_text)

    # Assert the questions are embedded in the full responses
    for message, response in zip(messages_batch, full_responses):
        assert message["question"] in response, f"Question not echoed in response"

    return full_responses


def process_model_output_batch_hf_endpoint(messages_batch, client, model_name):
    """Get model output for a batch of messages using HuggingFace Inference Providers.

    Args:
        messages_batch: List of message dictionaries
        client: InferenceClient instance
        model_name: Full model name (e.g., "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B")

    Returns:
        List of full responses (prompt + generated text)
    """
    # Build the model identifier with provider policy if specified
    if args.provider_policy != "auto":
        model_id = f"{model_name}:{args.provider_policy}"
    else:
        model_id = model_name

    full_responses = []

    # Process each message individually (HF Inference Providers doesn't batch internally)
    for message in messages_batch:
        # Build the chat messages format
        if args.is_base_model:
            # For base models, use the content directly as a single user message
            prompt = message["content"]

            try:
                # Use text generation instead of chat completion
                response = client.text_generation(
                    prompt,
                    model=model_id,
                    max_new_tokens=args.max_tokens,
                    temperature=args.temperature if args.temperature > 0 else None,
                    top_p=args.top_p if args.temperature > 0 else None,
                    do_sample=args.temperature > 0,
                )

                # Handle None response
                if response is None:
                    print(f"Warning: Received None response from text_generation API")
                    response = ""

                full_response = prompt + response

            except Exception as e:
                print(f"Error during text_generation API call: {e}")
                print(f"Model: {model_id}")
                print(f"Prompt length: {len(prompt)} chars")
                raise
        else:
            # For instruct models, use chat completion
            chat_messages = [{"role": message["role"], "content": message["content"]}]

            try:
                completion = client.chat.completions.create(
                    model=model_id,
                    messages=chat_messages,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature if args.temperature > 0 else None,
                    top_p=args.top_p if args.temperature > 0 else None,
                )

                # Reconstruct full response with prompt
                generated_text = completion.choices[0].message.content

                # Handle None response
                if generated_text is None:
                    print(f"Warning: Received None response from API. Full completion object: {completion}")
                    generated_text = ""

                full_response = message["content"] + "\n\n" + generated_text

            except Exception as e:
                print(f"Error during API call: {e}")
                print(f"Model: {model_id}")
                print(f"Message length: {len(message['content'])} chars")
                raise

        full_responses.append(full_response)

        # Assert the question is in the response (skip if empty response)
        if generated_text:
            assert message["question"] in full_response, f"Question not found in response"

    return full_responses


def process_messages(dataset_name, question_ids, messages_by_question_id, tokenizer, model, engine: str, remote: bool = False):
    """Process a batch of messages and return response data"""
    if engine == "vllm":
        assert args.batch_size >= 1, f"batch_size must be >= 1, got {args.batch_size}"

        all_data = []
        for start in tqdm(range(0, len(question_ids), args.batch_size)):
            sub_ids = question_ids[start:start + args.batch_size]
            messages_batch = [messages_by_question_id[qid] for qid in sub_ids]
            responses = process_model_output_batch_vllm(messages_batch, tokenizer, model)

            for message, response, question_id in zip(messages_batch, responses, sub_ids):
                all_data.append({
                    "original_message": {"role": message["role"], "content": message["content"]},
                    "full_response": response,
                    "question_id": question_id,
                    "category": message["category"],
                    "question": message["question"],
                    "gold_answer": message.get("gold_answer", ""),
                    "dataset_name": dataset_name
                })

            # Proactively release memory between micro-batches
            torch.cuda.empty_cache()
            gc.collect()

        return all_data
    elif engine == "nnsight":
        assert args.batch_size >= 1, f"batch_size must be >= 1, got {args.batch_size}"

        all_data = []
        for start in tqdm(range(0, len(question_ids), args.batch_size)):
            sub_ids = question_ids[start:start + args.batch_size]
            messages_batch = [messages_by_question_id[qid] for qid in sub_ids]
            responses = process_model_output_batch_nnsight(messages_batch, tokenizer, model, remote=remote)

            for message, response, question_id in zip(messages_batch, responses, sub_ids):
                all_data.append({
                    "original_message": {"role": message["role"], "content": message["content"]},
                    "full_response": response,
                    "question_id": question_id,
                    "category": message["category"],
                    "question": message["question"],
                    "gold_answer": message.get("gold_answer", ""),
                    "dataset_name": dataset_name
                })

            # Proactively release memory between micro-batches
            torch.cuda.empty_cache()
            gc.collect()

        return all_data
    elif engine == "hf_endpoint":
        assert args.batch_size >= 1, f"batch_size must be >= 1, got {args.batch_size}"

        all_data = []
        for start in tqdm(range(0, len(question_ids), args.batch_size)):
            sub_ids = question_ids[start:start + args.batch_size]
            messages_batch = [messages_by_question_id[qid] for qid in sub_ids]
            responses = process_model_output_batch_hf_endpoint(messages_batch, client=model, model_name=tokenizer)

            for message, response, question_id in zip(messages_batch, responses, sub_ids):
                all_data.append({
                    "original_message": {"role": message["role"], "content": message["content"]},
                    "full_response": response,
                    "question_id": question_id,
                    "category": message["category"],
                    "question": message["question"],
                    "gold_answer": message.get("gold_answer", ""),
                    "dataset_name": dataset_name
                })

        return all_data
    else:
        raise ValueError(f"Engine {engine} not supported")


def save_responses(responses_data, responses_json_path):
    with open(responses_json_path, "w") as f:
        json.dump(responses_data, f, indent=2)
    print(f"Saved {len(responses_data)} responses to {responses_json_path}") 


# Main execution
if __name__ == "__main__":
    model_name = args.model

    print(f"Loading model {model_name} using engine '{args.engine}'...")
    if args.engine == "vllm":
        if args.remote:
            raise ValueError("Remote execution not supported with vLLM engine, use --engine nnsight")
        # Lazy import to avoid vllm requirement for nnsight
        from vllm import LLM
        from transformers import AutoTokenizer
        model = LLM(
            model=model_name,
            tensor_parallel_size=args.tensor_parallel_size if args.tensor_parallel_size != -1 else torch.cuda.device_count(),
            dtype=args.dtype,
            seed=args.seed,
            max_model_len=args.max_tokens + MAX_TOKENS_IN_INPUT, # We assume inputs are 1000 tokens long at most
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    elif args.engine == "hf_endpoint":
        # Use HuggingFace Inference Providers
        from huggingface_hub import InferenceClient

        # Get API key from args or environment
        hf_token = args.api_key or os.environ.get("HF_TOKEN")
        if not hf_token:
            raise ValueError("HF token required for hf_endpoint engine. Set --api_key or HF_TOKEN environment variable")

        print(f"Using HuggingFace Inference Providers with policy: {args.provider_policy}")

        # Create the InferenceClient
        client = InferenceClient(token=hf_token)

        # For hf_endpoint, model is just the string name and client is the InferenceClient
        model = client  # Store client in model
        tokenizer = model_name  # Store model name in tokenizer (will be passed as string)
    else:
        # Use nnsight-based loading (supports both local and remote)
        model, tokenizer = utils.load_model(
            model_name=model_name,
            load_in_8bit=args.load_in_8bit,
            remote=args.remote,
            api_key=args.api_key
        )

    # Create directories
    os.makedirs('results/vars', exist_ok=True)

    model_prefix = "base_" if args.is_base_model else ""
    responses_json_path = f"results/vars/{model_prefix}responses_{model_name.split('/')[-1].lower()}.json"

    random.seed(args.seed)

    ds = load_dataset(args.dataset)
    rows = ds[args.dataset_split]
    messages_by_question_id = get_messages_from_dataset(args.dataset, rows)
    question_ids = list(messages_by_question_id.keys())

    random.shuffle(question_ids)

    # Limit to max_examples if specified
    if args.max_examples is not None:
        question_ids = question_ids[:args.max_examples]
        print(f"Processing {len(question_ids)} questions (limited by --max_examples) in {args.dataset_split} split of {args.dataset} to generate responses")
    else:
        print(f"Processing {len(question_ids)} questions in {args.dataset_split} split of {args.dataset} to generate responses")

    responses_data = process_messages(args.dataset, question_ids, messages_by_question_id, tokenizer, model, args.engine, remote=args.remote)
        
    # Clean up memory
    torch.cuda.empty_cache()
    gc.collect()

    # Save final results
    save_responses(responses_data, responses_json_path)