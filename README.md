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

Valerie's BPE tokenizer is currently **ASCII-only** for simplicity; Unicode (UTF-8 grapheme) support is planned.

**Pipeline:**

1. **Corpus aggregation:** Concatenate input text into a single file.
2. **Pre-tokenization:** Split text into tokens (whitespace or regex).
3. **Token frequency mapping:** Count occurrences of each token.
4. **Vocabulary build:** Construct a symbol table mapping tokens to frequencies.
5. **BPE training:** Iteratively merge the most frequent symbol pairs.
6. **Tokenizer model:** Construct the tokenizer model from the bpe merges.

**Typical Workflow:**

```sh
# Build base vocabulary from text
./build/examples/tokenizer/vocab --vocab samples/simple.txt

# Train BPE model with N merges
./build/examples/tokenizer/bpe --vocab samples/simple.txt --merges 10

# Build and serialize tokenizer model
./build/examples/tokenizer/model --input samples/simple.txt --output models
```

*Output:*

- Shows tokens and frequencies after each step.
- Prints BPE merge steps and selected pairs.

**Planned:**

- Final model is saved to the specified output directory.
- Full Unicode (grapheme) support.
- Model serialization, validation, and extensibility.

## License

AGPL to ensure end-user freedom.
