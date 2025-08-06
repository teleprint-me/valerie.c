"""
@file unicode.model.py
@ref https://aclanthology.org/P16-1162/
@ref https://aclanthology.org/2025.coling-main.400/
"""

import argparse
import collections
import json
import re


# @note The rationale for not using list() to split is because of the stop token.
#       If list were used, then the stop token would be split along with the rest of the string.
def corpus_default() -> list[str]:
    return ["lo", "low", "lower", "newest", "wide", "wider", "widest"]


def corpus_read(path: str) -> list[str]:
    words = []
    with open(path, "r") as file:
        corpus = file.read()
    lines = corpus.splitlines()
    for line in lines:
        for word in line.split():
            words.append(word)
    print(f"Initialized vocab from file: {path}")
    return words


def corpus_init(words: list[str]) -> dict[str, int]:
    vocab = {}
    for word in words:
        symbols = list(word)
        symbols.append("</w>")
        vocab[" ".join(symbols)] = 1
    print(json.dumps(vocab, indent=2))
    return vocab


def get_pairs(vocab: dict[str, int]) -> dict[tuple[str, str], int]:
    pairs = collections.defaultdict(int)  # init freqs to 0
    for word, freq in vocab.items():  # unpacks ("l o w </w>", 5)
        symbols = word.split()  # split word by char -> ["l", "o", "w", ...]
        for i in range(len(symbols) - 1):  # for each step in the set of symbols
            cur = symbols[i]  # "l"
            nxt = symbols[i + 1]  # "o"
            pairs[cur, nxt] += freq  # p[("l", "o")] += 1
            print(f"i={i}, cur='{cur}', nxt='{nxt}', freq={freq}")
    return pairs  # {('l', 'o'): 1}


def merge_pair(symbols: list[str], pair: tuple[str, str]) -> list[str]:
    out = []
    i = 0
    while i < len(symbols):
        # If this symbol and the next match the pair, merge them
        if i < len(symbols) - 1 and symbols[i] == pair[0] and symbols[i + 1] == pair[1]:
            out.append(symbols[i] + symbols[i + 1])
            i += 2  # Skip the next symbol (it's merged)
        else:
            out.append(symbols[i])
            i += 1
    return out


def get_merges(vocab: dict[str, int], pair: tuple[str, str]) -> dict[str, int]:
    print("Updated pairs:")
    print(json.dumps(vocab, indent=2))

    new_vocab = {}  # new empty vocab
    for word in vocab:  # for each pair in a given map
        symbols = word.split()  # ["l", "o", "w", "</w>"]
        merged = merge_pair(symbols, pair)
        new_word = " ".join(merged)
        print(f"word={word}, new_word={new_word}")
        new_vocab[new_word] = vocab[word]
    return new_vocab


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--merges", required=False, type=int, default=10)
    parser.add_argument("-c", "--corpus", required=False, type=str, default=None)
    args = parser.parse_args()

    words = corpus_default()
    if args.corpus:
        words = corpus_read(args.corpus)
    vocab = corpus_init(words)

    num_merges = int(args.merges)
    for i in range(num_merges):
        pairs = get_pairs(vocab)  # create pairs
        if not pairs:  # empty
            print(f"Exhausted all potential pairs! Halted at step {i}.")
            break
        best = max(pairs, key=pairs.get)  # get max rank
        print(f"best={best}")
        vocab = get_merges(vocab, best)  # merge ranked pair

    print("Final Best:")
    print(json.dumps(best, indent=2))
    print("Final Vocab:")
    print(json.dumps(vocab, indent=2))
