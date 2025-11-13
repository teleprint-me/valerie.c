# Valerie

Valerie is a Large Language Model written completely from scratch in pure C.

## Features

- [ ] UTF-8 grapheme support
- [ ] Byte-Pair Encoding (BPE) tokenizer
- [ ] Model weights: Q8 (inference), BF16 (training)
- [ ] File serialization & validation
- [ ] Completions engine
- [ ] Chat completions engine
- [ ] Training Engine
- [ ] Fine-tuning Engine
- [ ] Headless CPU Support (OpenMP)
- [ ] Headless GPU Support (Vulkan)

## Setup

```sh
git clone https://github.com/teleprint-me/valerie.c valerie
cd valerie
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j $(nproc)
```

## Tokenizer

Valerie includes an **ASCII-only Byte-Pair Encoding (BPE) tokenizer** designed for transparency and ease of extension. Unicode (UTF-8 grapheme) support is planned.

### Workflow

1. **Train the model:** Build and serialize a BPE tokenizer from a plaintext corpus.
2. **Predict:** Encode and decode text using a trained model.

### Commands

#### Train

Build and save a tokenizer model:

```sh
./build/examples/tokenizer/train --input S --output S [--merges N] [--verbose]
```

- `--input`, `-i`   Path to input plaintext corpus (required)
- `--output`, `-o`  Directory to save the tokenizer model (required)
- `--merges`, `-m`  Number of BPE merge steps (default: 10)
- `--verbose`, `-v` Enable debug output

#### Predict

Encode and decode text with a trained model:

```sh
./build/examples/tokenizer/predict --model S --prompt S [options]
```

- `--model`, `-m`   Path to tokenizer model file (required)
- `--prompt`, `-p`  Input text to encode and decode (required)
- `--add-bos`, `-b` Add BOS marker
- `--add-eos`, `-e` Add EOS marker
- `--verbose`, `-v` Enable debug output

### Example

**Train:**

```sh
./build/examples/tokenizer/train -i samples/simple.txt -o models -m 10
```

**Predict:**

```sh
./build/examples/tokenizer/predict -m models/tokenizer.model -p 'Hello, world!'
```

*Typical output:*

- Prints tokens, frequencies, and merge steps when training.
- Lists vocabulary and encodings when predicting.

**Planned:**

- Unicode grapheme support
- Model extensibility and validation

## Model

### What Is Valerie?

Valerie is a **decoder-only transformer** inspired by architectures like **GPT**, **Llama**, **Mistral**, and **Qwen**.
Its design closely follows [Adrian Cable’s Qwen3 C implementation](https://github.com/adriancable/qwen3.c/blob/main/runq.c), which provided an excellent reference for inference behavior. Valerie extends this concept beyond inference, toward a complete training and fine-tuning framework.

At its core, Valerie is an experiment in understanding and re-implementing large language model mechanics from first principles: every layer, tensor operation, and gradient is written manually, with full transparency and zero abstraction bloat.

### Why Build From Scratch?

I wanted to **understand** how a transformer truly works, not just use one.
That meant rebuilding every component from the ground up: tokenizer, model, optimizer, and serialization. Valerie depends only on minimal, transparent libraries like **PCRE2**, **OpenMP**, and (eventually) **Vulkan**, keeping the codebase small, portable, and easy to inspect.

Transformers are intricate systems grounded in **algebra**, **geometry**, **calculus**, and **statistics**. Each layer (attention, feed-forward, normalization) is a self-contained “computable block.” Valerie exposes these blocks directly, allowing the entire forward and backward pipeline to be followed line-by-line.

### Why Not PyTorch?

**PyTorch** is powerful but highly abstracted and optimized for NVIDIA hardware.
Its heavy CUDA focus, dependency footprint, and dynamic graph system hide too much of what I want to see, especially for low-level experimentation. While I appreciate Python’s flexibility, it isn’t well-suited for understanding the mechanics of transformers at the memory or numeric level.

By contrast, Valerie’s C implementation is **explicit and predictable**, running close to the metal and relying on a small, disciplined build.

### Why Not GGML?

**GGML** is an excellent inference framework supporting many architectures.
However, its computation-graph-based design (a Directed Acyclic Graph or **DAG**) makes it difficult to trace the fundamental operations without stepping through layers of abstraction. Valerie takes the opposite approach: a **linear, transparent, and manually written** execution path that prioritizes understanding over optimization.

### Why in C?

C offers the right balance of **simplicity, speed, and control**.
There’s no hidden allocation, no garbage collector, and no surprise abstractions, just raw access to the system. That control comes at a cost: safety and patience are required.
Valerie relies on **AddressSanitizer (ASAN)** during development to catch common memory issues early, but careful engineering discipline remains essential.

C has been my language of choice for years, and Valerie reflects my belief that with care, C can still serve as a foundation for modern, high-performance machine learning research.

### Current Status

Valerie is a **work in progress**.
The architecture, forward pass, and training loop are nearly complete, but issues remain in three main areas:

* **Initialization** – potential scaling or variance imbalance
* **Gradient accumulation** – instability during backpropagation
* **Buffer management** – cleanup and consistency for precision variants

Currently, gradients tend to explode, preventing the model from converging or generalizing. Single-precision (FP32) debugging is the primary focus before expanding to mixed and quantized formats.

You can view the model implementation here:
[`examples/model/v.c`](examples/model/v.c)

The code runs, but training stability is still under investigation.

### Contributions

Contributions are welcome, clarity and simplicity are the guiding principles.
Before optimizing for performance, Valerie aims to **work correctly, read clearly, and explain itself**.

## License

AGPL to ensure end-user freedom.
