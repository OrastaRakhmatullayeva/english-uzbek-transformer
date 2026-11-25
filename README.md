
---
title: LingoUz
emoji: 🌐
colorFrom: blue
colorTo: purple
sdk: gradio
app_file: app.py
pinned: false
---

# LingoUz — English-to-Uzbek Transformer

A character-level neural word translation system built
with PyTorch, inspired by the Transformer architecture
introduced in "Attention Is All You Need" (2017).

## Project Overview

LingoUz explores neural machine translation for Uzbek,
a low-resource language in natural language processing.

The model is trained from scratch without relying on
pretrained language models or translation APIs.

## Tech Stack

- Python
- PyTorch
- Transformer Architecture
- Natural Language Processing
- Gradio

## Model Architecture

| Parameter | Value |
|---|---|
| Architecture | Encoder-Decoder Transformer |
| Tokenization | Character-level |
| Embedding dimension | 64 |
| Attention heads | 4 |
| Encoder layers | 2 |
| Decoder layers | 2 |
| Feed-forward dimension | 128 |
| Dropout | 0.1 |
| Decoding | Autoregressive greedy decoding |

## Dataset

The initial dataset contains 100 English-Uzbek word pairs.

Dataset location:
`data/english_uzbek_100.csv`

This dataset is intended for educational experiments,
not production-level machine translation.

## Key Features

- Custom encoder-decoder architecture
- Sinusoidal positional encoding
- Multi-head self-attention
- Encoder-decoder cross-attention
- Padding and causal attention masks
- Autoregressive inference
- Interactive Gradio interface

## Installation

```bash
git clone https://github.com/OrastaRakhmatullayeva/english-uzbek-transformer.git
cd english-uzbek-transformer
pip install -r requirements.txt
```

Place the trained model weights and matching vocabulary
files inside the `checkpoints/` directory.

Run the application:

```bash
python app.py
```

## Project Structure

```text
english-uzbek-transformer/
├── checkpoints/
├── data/
├── model.py
├── inference.py
├── app.py
├── requirements.txt
└── README.md
```

## Evaluation

Quantitative evaluation results will be added after
running reproducible evaluation experiments.

## Limitations

- Small training dataset
- Word-level translation only
- Limited generalization to unseen inputs
- Greedy decoding
- Not intended for production use

## Future Improvements

- Expand the parallel corpus
- Support sentence-level translation
- Add reproducible training and evaluation pipelines
- Compare alternative model configurations
- Develop a REST API
- Optimize inference performance

## Reference

Vaswani et al. (2017).
[Attention Is All You Need](https://arxiv.org/abs/1706.03762).

## Author

Orasta Rakhmatullayeva
