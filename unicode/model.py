"""
@file unicode.model.py
@ref https://aclanthology.org/P16-1162/
@name Algorithm 1 Learn BPE operations
"""

import argparse
import collections
import json
import re


def corpus_default() -> dict[str, int]:
    return {
        "l o </w>": 1,
        "l o w </w>": 1,
        "l o w e r </w>": 1,
        "n e w e s t </w>": 1,
        "w i d e </w>": 1,
        "w i d e r </w>": 1,
        "w i d e s t </w>": 1,
    }


def corpus_read(path: str) -> dict[str, int]:
    vocab = {}
    with open(args.corpus, "r") as file:
        corpus = file.read()
    lines = corpus.splitlines()
    for line in lines:
        for word in line.split():
            symbols = list(word)
            symbols.append("</w>")
            vocab[" ".join(symbols)] = 1
    print(f"Initialized vocab from file: {args.corpus}")
    print(json.dumps(vocab, indent=2))
    return vocab


def get_pairs(vocab: dict[str, int]) -> dict[tuple[str, ...], int]:
    pairs = collections.defaultdict(int)  # init freqs to 0
    for word, freq in vocab.items():  # unpacks ("l o w </w>", 5)
        symbols = word.split()  # split word by char -> ["l", "o", "w", ...]
        for i in range(len(symbols) - 1):  # for each step in the set of symbols
            cur = symbols[i]  # "l"
            nxt = symbols[i + 1]  # "o"
            pairs[cur, nxt] += freq  # p[("l", "o")] += 1
            print(f"i={i}, cur='{cur}', nxt='{nxt}', freq={freq}")
    return pairs  # {('l', 'o'): 1}


def get_merges(vocab: dict[tuple[str, ...]], pair: str):
    print("Updated pairs:")
    print(json.dumps(vocab, indent=2))

    new_vocab = {}  # new empty vocab
    bigram = re.escape(" ".join(pair))  # Escape spaces: ('l', 'o') -> "l\ o"
    # Regex: Match only 'l o' as whole tokens, not inside others (word-boundaries)
    # (?<!\S): Not preceded by non-whitespace (i.e., start or whitespace)
    # (?!\S): Not followed by non-whitespace (i.e., end or whitespace)
    match = re.compile(r"(?<!\S)" + bigram + r"(?!\S)")
    print(f"pair={pair}, bigram={bigram}, match={match}")
    for word in vocab:  # for each pair in a given map
        # Replace the bigram in the word (all occurrences)
        new_word = match.sub("".join(pair), word)
        print(f"word={word}, new_word={new_word}")
        new_vocab[new_word] = vocab[word]
    return new_vocab


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--merges", required=False, type=int, default=10)
    parser.add_argument("-c", "--corpus", required=False, type=str, default=None)
    args = parser.parse_args()

    vocab = corpus_default()
    if args.corpus:
        vocab = corpus_read(args.corpus)

    num_merges = int(args.merges)
    for i in range(num_merges):
        pairs = get_pairs(vocab)
        if not pairs:
            print(f"Exhausted all potential pairs! Halted at step {i}.")
            break
        best = max(pairs, key=pairs.get)
        vocab = get_merges(vocab, best)

    print("Best:")
    print(json.dumps(best, indent=2))
    print("Vocab:")
    print(json.dumps(vocab, indent=2))
