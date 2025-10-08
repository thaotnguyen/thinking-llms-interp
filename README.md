# Base Models Know How to Reason, Thinking Models Learn When

Code for the paper [Base Models Know How to Reason, Thinking Models Learn When](https://openreview.net/forum?id=12Tc0R0I8j).

**Website:** [thinking-llms-interp.com](https://thinking-llms-interp.com/)

## Setup

### Requirements

- Python 3.10+
- `uv` installed (`pip install uv` or see the [uv docs](https://docs.astral.sh/uv/getting-started/installation/))

### Install

```bash
git clone https://github.com/cvenhoff/cot-interp.git
cd cot-interp
uv sync
```

## Generating thinking model responses

To generate responses from thinking models on MMLU-Pro:

```bash
cd generate-responses
./run.sh
```

This will generate responses from multiple thinking models (DeepSeek-R1 variants and QwQ) with their reasoning traces.

## Training taxonomy

To train and evaluate taxonomies:

```bash
cd train-saes
./run.sh
```

This will:
- Collect activations for each of the selected layers for each model
- Train all the Sparse Autoencoders (SAEs) for different cluster sizes, for each selected layer on each model
- Generate titles and descriptions for each cluster in the trained SAEs (5 repetitions by default)
- Evaluate all the candidate taxonomies (using the 5 default repetitions if available)
- Plot results

## Annotating thinking traces

To annotate thinking traces using a given taxonomy (specific layer and cluster size):

```bash
cd generate-responses
./run_annotation.sh
```

This will annotate the thinking traces for each model using the selected taxonomy.

## Training steering vectors

To train steering vectors for the models used in the paper:

```bash
cd train-vectors

# For each model, run the corresponding script:
./run_qwen_1.5b.sh
./run_llama_8b.sh
./run_qwen_14b.sh
./run_qwen_32b_linear_on_deepseek.sh
./run_qwen_32b_linear_on_qwq.sh
```

## Running hybrid model

To run hybrid model experiments:

```bash
cd hybrid-2

# Run experiments for different models:
./run_qwen_1.5b.sh
./run_llama_8b.sh
./run_qwen_14b.sh
./run_qwen_32b_on_deepseek.sh
./run_qwen_32b_on_qwq.sh

# Additional ablation experiments:
./run_qwen_32b_only_bias.sh
./run_qwen_32b_random_firing.sh
./run_qwen_32b_random_vectors.sh
```

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{venhoff2025thinking,
  title={Base Models Know How to Reason, Thinking Models Learn When},
  author={Venhoff, Constantin and Arcuschin, Iv{\'a}n and Torr, Philip and Conmy, Arthur and Nanda, Neel},
  booktitle={Mechanistic Interpretability Workshop at NeurIPS},
  year={2025}
}
```
