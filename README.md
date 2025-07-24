# Valerie

Valerie is a Large Language Model written completely from scratch in pure C.

## Features

- [ ] Full UTF-8 Support.
- [ ] Byte Pair Encoding Tokenizer.
- [ ] Model Weights in Q8 (inference) and BF16 (training).
- [ ] File Serialization and Validation.
- [ ] Completions Engine.
- [ ] Chat Completions Engine.
- [ ] Training Engine.
- [ ] Fine-tuning Engine.
- [ ] Headless CPU Support (OpenMP).
- [ ] Headless GPU Support (Vulkan).

## Setup

```sh
git clone https://github.com/teleprint-me/valerie.c valerie
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j $(nproc)
```

## License

AGPL to ensure end-user freedom.
