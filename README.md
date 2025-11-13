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

### What is Valerie?

Valerie is a derivation of GPT, Llama, Mistral, and Qwen.

Valeries architecture primarily mirrors the Qwen3 model which is based off of [Adrian Cable's Qwen3 implementation in C](https://github.com/adriancable/qwen3.c/blob/main/runq.c). While this is really cool and I had a lot of fun playing with the code, it is inference only.

As a result, Valerie is a decoder-only transformer model for generating and predicting natural language.

### Why not Transformers and Tokenizers?

I wanted to learn about and comprehend how a language model operates from the ground up. This meant rewriting the majority of the code because I had to consider every component from inference to back-propagation. This meant writing not only the tokenizer completely from scratch, but the model itself as well. As a result, Valerie has no external dependencies aside from a few helper libraries like PCRE2, OpenMP, and Vulkan (planned, but not implemented for simplicity).

Modern architectures are extremely complicated and have a lot of moving parts. The forward pass computes the predictions while the backward pass computes the errors within those predictions. This is a combination of Algebra (Basic, Set, and Discrete), Geometry, Trigonometry, Calculus, and Statistics. The models are composed of computable units which could be referenced as "blocks". Each block accepts an input and produces an output.

### Why not PyTorch?

PyTorch mostly focuses on CUDA. NVIDIA GPU's are expensive and AMD GPU's are cheaper, but ROCm has a lot left to be desired, which is unfortunate. PyTorch is extremely slow on operations which are typically O(n^3) at worst with the attention mechanism being a quadradic complexity and has limted support as a result. Since PyTorch focuses on specific hardward vendors, the code is not as portable or flexible. I love Python as a language, but it is far from perfect. I have years of experience using this language, and the dependencies for PyTorch can end up being gigabytes in size. This doesn't include the added libraries from Meta and HuggingFace.

### Why not GGML?

GGML is primarily an inference engine and does support optimization, but is incredibly complex due to the fact that it supports many architectures. This is not a bad thing. In fact, I think there are many amazing libraries that exist, but they all use something known as a Computation Graph, (aka a Directed Acyclic Graph or DAG for short). As a result, the DAG ends up obfuscating a lot of the core operations. C++ is also an incredibly complex language. There are multiple ways to do the same thing and I often find the features of the language to be more of a burden than a gift.

### Why in C?

I quite like C. It's simple, fast, and gives users complete control. I have no problem filling in the gaps of the language. While memory safety is not as robust as other languages, it is a language that I prefer due to years of familiarity with it. C was my first real programming language as I only had exposure to scripting languages like PHP, Python, and JavaScript up until that point. Employing ASAN in the development build helps catch common issues early, though, it is far from perfect and does not always work. As a result, C requires equal levels of discipline, patience, and vigilance.

### Where can I find the model?

Valerie is currently a work in progress. The model is nearly complete, but there are some outstanding issues I'm currently facing. I've narrowed it down to 3 potential areas: Initialization, gradient accumulation, and buffer management. Buffer management is a low priority until single precision is completely ironed out. Currently, the gradients explode, which leads to the model failing to learn or generalize. There are a bunch of reasons why this might be the case and I'm currently working towards a solution.

You can [see the model here](examples/model/v.c). The code does execute, but the model fails to learn.

### Are you open to contributions?

Yes. I am open to contributions. Solutions must be simple, transparent, and clear. I prioritize code clarity above all else. I usually consider optimization last. Currently, basic functionality is a priority for me. 

## License

AGPL to ensure end-user freedom.
