"""
@file unicode.model.py
@ref https://aclanthology.org/P16-1162/
@name Algorithm 1 Learn BPE operations
"""

import argparse
import collections
import json
import re


def get_stats(vocab):
    pairs = collections.defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[symbols[i], symbols[i + 1]] += freq
    return pairs


def merge_vocab(pair, v_in):
    v_out = {}
    bigram = re.escape(" ".join(pair))
    p = re.compile(r"(?<!\S)" + bigram + r"(?!\S)")
    for word in v_in:
        w_out = p.sub("".join(pair), word)
        v_out[w_out] = v_in[word]
    return v_out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--merges", required=False, default=10)
    args = parser.parse_args()

    vocab = {
        "l o w </w>": 5,
        "l o w e r </w>": 2,
        "n e w e s t </w>": 6,
        "w i d e s t </w>": 3,
    }

    num_merges = int(args.merges)
    for i in range(num_merges):
        pairs = get_stats(vocab)
        if not pairs:
            print(f"Exhausted all potential pairs! Halted at step {i}.")
            break
        best = max(pairs, key=pairs.get)
        vocab = merge_vocab(best, vocab)

    print("Best:")
    print(json.dumps(best, indent=2))
    print("Vocab:")
    print(json.dumps(vocab, indent=2))
