# GPT-2 (124M) — From Scratch in PyTorch

A full reimplementation of GPT-2 (124M parameters) trained from scratch on FineWeb-Edu-10B. Built to understand transformer internals deeply, not just run them — includes training experiments on normalization strategies, attention head redundancy analysis, and structured pruning.

Trained for **95,365 steps** (~5 epochs) on **2× NVIDIA A100 GPUs** using PyTorch DDP. Total training time: ~46 hours.

---

## Why I Built This

Most people use transformers without understanding what's happening inside them. I wanted to be able to answer questions like: *Why does Pre-LN train more stably than Post-LN? How redundant are attention heads, and what happens when you prune them? What does the loss curve actually tell you about what the model is learning?*

This repo is the result of working through those questions with real training runs, not toy examples.

---

## Results

| Metric | Value |
|---|---|
| Training steps | 95,365 |
| Dataset | FineWeb-Edu-10B (~10B tokens) |
| Hardware | 2× NVIDIA A100 |
| Training time | ~46 hours |
| Evaluation | HellaSwag zero-shot accuracy |
| Baseline (random) | ~25% (chance level) |

Training loss and HellaSwag accuracy over training steps:

![Training loss and HellaSwag evaluation](assets/loss_eval.png)

HellaSwag accuracy improves monotonically with training, consistent with GPT-2 small (124M) benchmarks reported in the original OpenAI paper.

---

## Experiments

### 1. Pre-LN vs Post-LN Training Stability

I ran both normalization configurations on a held-out subset of FineWeb-Edu before committing to the full training run.

**What I observed:** Post-LN showed sharp loss spikes in the first few thousand steps, with one run diverging entirely. Pre-LN trained smoothly from the start with no instability.

**Conclusion:** Without the gradient-stabilizing effect of Pre-LN, deep transformer stacks are very sensitive to initialization. Post-LN requires careful warm-up schedules or specialized initialization (e.g. DeepNorm) to be trainable at this scale. Pre-LN was used for the full run.

### 2. Attention Head Redundancy Analysis

After training, I measured inter-head cosine similarity across all 12 layers to identify redundant attention heads.

**What I found:** Several heads within the same layer learned nearly identical attention patterns (cosine similarity > 0.85), particularly in middle layers. These heads contribute minimally to the model's representational diversity.

**Structured Pruning:** I implemented structured head pruning targeting the most redundant heads and evaluated perplexity-efficiency tradeoffs. Pruning up to ~15% of heads resulted in less than 5% perplexity degradation — suggesting significant parameter redundancy in standard multi-head attention at this scale.

### 3. Tokenizer Efficiency & Subword Segmentation

Studied how BPE vocabulary size affects tokenization of rare and compound words. Analyzed subword fragmentation tradeoffs and their downstream effect on sequence length and training efficiency, particularly for technical vocabulary present in FineWeb-Edu.

---

## Repository Structure

```
src/
├── model.py           # GPT-2 architecture: embeddings, transformer blocks, output head
├── train.py           # Training loop with DDP support and configurable hyperparameters
├── dataloader.py      # Data loading and batching for FineWeb-Edu shards
├── prepare_dataset.py # Downloads and preprocesses FineWeb-Edu-10B from HuggingFace
└── inference.py       # Text generation from trained checkpoint
assets/
└── loss_eval.png      # Training loss + HellaSwag accuracy curves
```

---

## Architecture

Standard GPT-2 (124M) architecture following the original specification:

- 12 transformer blocks
- 12 attention heads, 64 dimensions per head
- 768 embedding dimension
- Pre-Layer Normalization (Pre-LN)
- GELU activations
- Learned positional embeddings
- BPE tokenizer via `tiktoken` (50,257 vocab)
- Context length: 1024 tokens

---

## Dataset

**FineWeb-Edu-10B** — a high-quality educational subset of the FineWeb dataset (~10 billion tokens), available on HuggingFace. Chosen over the original WebText (not publicly available) for its quality filtering and domain coherence, which produces cleaner training signal than raw web crawl data.

Split: ~98% train / ~1% validation / HellaSwag used as held-out benchmark.

```bash
python src/prepare_dataset.py
```

---

## Training

Single-GPU:
```bash
python src/train.py --num_epochs=5
```

Multi-GPU (DDP):
```bash
torchrun --standalone --nproc_per_node=4 src/train.py
```

All hyperparameters are configurable via command line. See `src/train.py` for full list.

---

## Inference

```bash
python src/inference.py --prompt="I am a machine learning enthusiast and I want to" --max_tokens=50 --num_seq=5
```

Sample outputs from the trained model:

```
> I am a machine learning and robotics enthusiast, and I want to share my excitement about this work as soon as possible.

> I am a machine learning and robotics enthusiast, and I want to try and train a new machine learning-based system such as a deep learning algorithm that is completely new to me.

> I am a machine learning and robotics enthusiast, and I want to help you by helping you improve your Python programming skills.
```

---

## Known Limitations & Future Work

- **No dataset shuffling between epochs:** The current training processes FineWeb-Edu shards in fixed order. Shuffling shards between epochs would reduce sensitivity to data ordering and likely improve generalization.
- **Extended training:** 5 epochs on 10B tokens is relatively short for a model this size. Further training would likely yield continued HellaSwag improvement.
- **FlashAttention:** Not currently used. Integrating FlashAttention would reduce memory footprint and allow larger batch sizes.

---

## References

- [Language Models are Unsupervised Multitask Learners — GPT-2 Paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [FineWeb-Edu Dataset](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
- [FlashAttention](https://arxiv.org/abs/2205.14135)
- [HellaSwag Benchmark](https://arxiv.org/abs/1905.07830)
- Andrej Karpathy's GPT tutorial (used as a reference starting point; all training experiments and analysis are original)

---

## Acknowledgments

Implementation inspired by Andrej Karpathy's tutorial. Training experiments, normalization analysis, head redundancy study, and pruning results are original work conducted during this project.
