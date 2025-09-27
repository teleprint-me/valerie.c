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

* `--input`, `-i`   Path to input plaintext corpus (required)
* `--output`, `-o`  Directory to save the tokenizer model (required)
* `--merges`, `-m`  Number of BPE merge steps (default: 10)
* `--verbose`, `-v` Enable debug/verbose output

#### Predict

Encode and decode text with a trained model:

```sh
./build/examples/tokenizer/predict --model S --prompt S [options]
```

* `--model`, `-m`   Path to tokenizer model file (required)
* `--prompt`, `-p`  Input text to encode and decode (required)
* `--add-bos`, `-b` Add BOS marker
* `--add-eos`, `-e` Add EOS marker
* `--verbose`, `-v` Enable debug output

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

* Prints tokens, frequencies, and merge steps when training.
* Lists vocabulary and encodings when predicting.

**Planned:**

* Unicode grapheme support
* Model extensibility and validation

## License

AGPL to ensure end-user freedom.
