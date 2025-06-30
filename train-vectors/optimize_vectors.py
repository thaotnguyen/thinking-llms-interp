# %%
import dotenv
dotenv.load_dotenv("../.env")

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from utils import steering_opt

# %%
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-1.5B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Math-1.5B").to(dtype=torch.bfloat16) # load in bfloat16 to use less VRAM

# %%
device = 'cuda'
torch.set_default_device(device)

model = model.to(device=device)

# %%
prompt = "Question: What comes next: Ra, Ac, Tu, Wd, Th, _____\nAnswer: "

# %%
generated_tokens = model.generate(**tokenizer(prompt, return_tensors='pt'), max_new_tokens=15)
generated_str = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0].replace(prompt, "")
print(generated_str)
# %%
og_completion = """09

Explanation: The sequence represents the days of the week, starting"""

target_completion = """I need to determine the next item in the sequence Ra, Ac, Tu, Wd, Th, and then figure out what comes after that."""
# %%
datapoint = steering_opt.TrainingDatapoint(
    prompt,
    src_completions=[og_completion], # src_completions: list of completions whose probability we want to decrease
    dst_completions=[target_completion], # dst_completions: list of completions whose probability we want to increase
)

# %%
datapoints = [datapoint] # a list of datapoints to optimize on; for now, only one datapoint
layer = 8 # the layer that we want to steer at

vector, loss_info = steering_opt.optimize_vector(
    model, datapoints, layer,
    tokenizer=tokenizer, # for HuggingFace models, we have to pass the tokenizer as well
    max_iters=20, # stop after 20 optimization iterations
    lr=0.1 # set the optimizer learning rate; by default, it's 0.01
)


# %%
print(loss_info)


# %%
steering_hook = (layer, steering_opt.make_steering_hook_hf(vector))

with steering_opt.hf_hooks_contextmanager(model, [steering_hook]): 
    # generate a steered completion
    generated_tokens = model.generate(**tokenizer(prompt, return_tensors='pt'), max_new_tokens=30)

generated_str = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

# %%
print(generated_str)

# %%
prompt2 = """What shape do you get when you slice a tetrahedron parallel to one of its faces?"""
max_new_tokens = 35

print("--- Unsteered generation ---")
generated_tokens = model.generate(**tokenizer(prompt2, return_tensors='pt'), max_new_tokens=max_new_tokens)
generated_str = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
print(generated_str)
print()

print("--- Steered generation ---")
steering_hook = (layer, steering_opt.make_steering_hook_hf(vector))
with steering_opt.hf_hooks_contextmanager(model, [steering_hook]): 
    generated_tokens = model.generate(**tokenizer(prompt2, return_tensors='pt'), max_new_tokens=max_new_tokens)
    generated_str = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    print(generated_str)

# %%
