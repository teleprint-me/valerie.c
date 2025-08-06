"""
@file unicode.model.py
@ref https://aclanthology.org/P16-1162/
@ref https://aclanthology.org/2025.coling-main.400/
"""

import argparse
import collections
import json
import math


# @note The rationale for not using list() to split is because of the stop token.
#       If list were used, then the stop token would be split along with the rest of the string.
def corpus_default() -> list[str]:
    print("Using default corpus.")
    return ["lo", "low", "lower", "newest", "wide", "wider", "widest"]


def corpus_read(path: str) -> list[str]:
    """Load a flat list of words from a file, one per whitespace."""
    words = []
    with open(path, "r") as file:
        for line in file:
            for word in line.split():
                words.append(word)
    print(f"Using corpus from file: {path}")
    return words


def corpus_init(words: list[str], stop_token="</w>") -> dict[str, int]:
    """Convert list of words into vocab dict: space-joined symbols (with stop token) → freq."""
    vocab = {}
    for word in words:
        symbols = list(word)
        symbols.append(stop_token)
        vocab[" ".join(symbols)] = 1
    print("Initialized vocab:")
    print(json.dumps(vocab, indent=2))
    return vocab


def get_pairs(vocab: dict[str, int]) -> dict[tuple[str, str], int]:
    print("Generating pairs:")
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--merges",
        required=False,
        type=int,
        default=10,
        help="number of merges",
    )
    parser.add_argument(
        "-c",
        "--corpus",
        required=False,
        type=str,
        default=None,
        help="input plaintext file",
    )
    parser.add_argument(
        "-e",
        "--eos",
        required=False,
        type=str,
        default="</w>",
        help="end-of-sequence token",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Get words from corpus (training data)
    words = None
    if args.corpus:
        words = corpus_read(args.corpus)
    else:
        words = corpus_default()

    # Get number of merges (training cycles)
    num_merges = int(args.merges)

    # Train vocab model (vocab is the set of all merges)
    vocab = corpus_init(words, args.eos)
    for i in range(num_merges):
        # pre-process merge pairs every cycle
        pairs = get_pairs(vocab)  # create pairs
        if not pairs:  # bail if pairs is empty
            print(f"Exhausted all potential pairs! Halted at step {i}.")
            break
        # use the highest ranked pair for the next merge cycle
        best = max(pairs, key=pairs.get)  # get max rank
        vocab = get_merges(vocab, best)  # merge ranked pair

    # Print vocab training results (dump merges)
    print("Final Best:")
    print(json.dumps(best, indent=2))
    print("Final Vocab:")
    print(json.dumps(vocab, indent=2))

    # Build the rank table (rank merges)
    rank_table = {}
    for i, merge in enumerate(vocab.keys()):
        pair = merge.split()
        token = "".join(pair)
        rank_table[token] = i
        print(f"merge={merge}, pair={pair}, token={token}, rank={i}")

    # Score the merges
    scores = {}
    for merge in vocab.keys():
        rank = rank_table.get(merge)
        scores[merge] = -math.log(rank + 1) if rank else -1e6
        print(f"merge={merge}, rank={rank}, score={scores[merge]}")

    # Collect All Unique Tokens (order matters!)
    # For every key, split by space, add each symbol to a set.
    token_set = set()
    for word in vocab:
        for symbol in word.split():
            token_set.add(symbol)

    # Assign IDs
    token_list = list(token_set)  # or preserve merge order

    # Map each unique token (symbol) to an integer ID.
    token_to_id = {token: idx for idx, token in enumerate(token_list)}
    id_to_token = {idx: token for idx, token in enumerate(token_list)}

    print("Tokenizer:")
    for token, id in token_to_id.items():
        print(f"token={token}, id={id}")
