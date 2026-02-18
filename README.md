<div align="center">

# 🇮🇳 BharatLLM

### A 7-Billion Parameter Hindi-English-Hinglish Language Model, Trained from Scratch

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org/)
[![Stars](https://img.shields.io/github/stars/chin-cyber21/bharatllm?style=social)](https://github.com/chin-cyber21/bharatllm)

*Not fine-tuned. Not quantized from another model. Built from scratch.*

</div>

---

## Why BharatLLM?

Most Hindi AI models are English models with Hindi fine-tuning bolted on. They carry English-first tokenization, English-first attention patterns, and fundamentally think in English before translating.

BharatLLM is different. It was built from the ground up on a 45TB+ corpus of Hindi, English, and Hinglish text — the way Indians actually communicate. The tokenizer understands Devanagari natively. The model thinks in Hindi first.

**Result**: 18% improvement on Indic language benchmarks compared to existing models of similar size.

---

## Architecture

```
Model Type:         Decoder-only Transformer (GPT-style)
Parameters:         7 Billion
Training Hardware:  8× NVIDIA A100 80GB GPUs
Training Strategy:  PyTorch FSDP (Fully Sharded Data Parallel)
Tokenizer:          Custom BPE — Hindi/English/Hinglish-aware
Context Length:     4096 tokens
Attention:          Multi-head self-attention with RoPE positional encodings
```

### What "from scratch" actually means

- Custom tokenizer trained on Indic corpus (not borrowed from LLaMA or GPT)
- Custom positional encodings with Rotary Position Embedding (RoPE)
- Custom attention implementation with Flash Attention optimization
- Custom FSDP sharding strategy across 8 A100s
- No LoRA. No QLoRA. No fine-tuning of existing checkpoints.

---

## Data Pipeline

```
Raw Sources → Apache Spark → Quality Filters → Deduplication → Tokenization → Training
    45TB+         PySpark         Custom rules      MinHash LSH      BPE (custom)
```

**Data sources included:**
- Hindi Wikipedia + Wikisource
- Common Crawl (Hindi + Hinglish filtered)
- Sangraha dataset (curated Indic text)
- Custom scraped Indian news corpora
- Code-switched (Hinglish) social media text

---

## Inference Performance

| Metric | Value |
|--------|-------|
| Throughput | 100+ requests/second |
| P99 Latency | 120ms |
| Batch Strategy | Dynamic batching via vLLM |
| Optimization | TensorRT + CUDA kernel fusion |

```bash
# Quick start (inference)
pip install vllm
python -m vllm.entrypoints.openai.api_server \
  --model chin-cyber21/bharatllm-7b \
  --dtype float16 \
  --max-model-len 4096
```

---

## Benchmarks

| Benchmark | BharatLLM-7B | LLaMA-2-7B | Bloom-7B |
|-----------|:---:|:---:|:---:|
| IndicGLUE (Hindi) | **71.4** | 60.2 | 58.8 |
| Indic Sentiment | **84.1** | 71.3 | 69.9 |
| Hindi QA | **68.7** | 55.1 | 53.4 |
| Hinglish NLI | **72.3** | 61.8 | 59.2 |

*18% average improvement on Indic language tasks vs comparable parameter-count models.*

---

## Repository Structure

```
bharatllm/
├── tokenizer/
│   ├── train_tokenizer.py      # BPE tokenizer training on Indic corpus
│   ├── tokenizer_config.json
│   └── vocab.json
├── model/
│   ├── architecture.py         # Full transformer implementation from scratch
│   ├── attention.py            # Multi-head attention + RoPE
│   ├── fsdp_config.py          # 8× A100 FSDP sharding strategy
│   └── config.py               # Model hyperparameters
├── data/
│   ├── pipeline.py             # Apache Spark data processing pipeline
│   ├── quality_filter.py       # Custom quality filters for Hindi/Hinglish
│   └── dedup.py                # MinHash LSH deduplication
├── training/
│   ├── train.py                # Main training loop
│   ├── scheduler.py            # Cosine LR with warmup
│   └── checkpointing.py        # Distributed checkpoint management
├── inference/
│   ├── serve.py                # vLLM inference server
│   └── benchmark.py            # Latency/throughput benchmarking
└── eval/
    └── indic_eval.py           # IndicGLUE benchmark evaluation
```

---

## Training Details

```python
# Model Configuration
config = BharatLLMConfig(
    vocab_size=64000,        # Larger vocab for Hindi/Hinglish coverage
    hidden_size=4096,
    num_hidden_layers=32,
    num_attention_heads=32,
    intermediate_size=11008,
    max_position_embeddings=4096,
    rope_theta=10000.0,
)

# FSDP Training (8× A100)
# Global batch size: 4M tokens
# Learning rate: 3e-4 with cosine decay
# Warmup steps: 2000
# Training tokens: ~1T
```

---

## Why This Matters for India

700M+ Hindi speakers. 1.4B people who code-switch between Hindi and English daily. The models they use were built for English first. BharatLLM is a step toward fixing that.

---

## Citation

```bibtex
@misc{saraswat2024bharatllm,
  author = {Chirag Saraswat},
  title = {BharatLLM: A 7B Parameter Hindi-English-Hinglish Language Model},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/chin-cyber21/bharatllm}
}
```

---

## About the Author

Built by [Chirag Saraswat](https://github.com/chin-cyber21), AI Architect [Codmek Softech](https://codmek.com).

*Questions? Open a Discussion. Want to contribute? PRs welcome.*
