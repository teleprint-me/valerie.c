"""
@file unicode.model.py
@ref https://aclanthology.org/P16-1162/
@ref https://aclanthology.org/2025.coling-main.400/
"""

import argparse
import collections
import json
import math


class Corpus:
    """Load and initialize training data"""

    @staticmethod
    def default() -> list[str]:
        return ["lo", "low", "lower", "newest", "wide", "wider", "widest"]

    @staticmethod
    def read(path: str) -> list[str]:
        """Load a flat list of words from a file, one per whitespace."""
        words = []
        with open(path, "r") as file:
            for line in file:
                for word in line.split():
                    words.append(word)
        return words

    @staticmethod
    def words(path: str = None) -> list[str]:
        if path:
            print(f"Using corpus from file: {path}")
            return Corpus.read(path)
        print("Using default corpus.")
        return Corpus.default()

    @staticmethod
    def vocab(path: str = None) -> dict[str, int]:
        """Convert list of words into vocab dict: space-joined symbols -> freq."""
        vocab = {}
        for word in Corpus.words(path):
            symbols = list(word)
            vocab[" ".join(symbols)] = 1
        print("Initialized vocab:")
        print(json.dumps(vocab, indent=2))
        return vocab


def get_pairs(vocab: dict[str, int]) -> dict[tuple[str, str], int]:
    # print("Generating pairs:")
    pairs = collections.defaultdict(int)  # init freqs to 0
    for word, freq in vocab.items():  # unpacks ("l o w </w>", 5)
        symbols = word.split()  # split word by char -> ["l", "o", "w", ...]
        for i in range(len(symbols) - 1):  # for each step in the set of symbols
            cur = symbols[i]  # "l"
            nxt = symbols[i + 1]  # "o"
            pairs[cur, nxt] += freq  # p[("l", "o")] += 1
            # print(f"i={i}, cur='{cur}', nxt='{nxt}', freq={freq}")
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
    # print("Updated pairs:")
    # print(json.dumps(vocab, indent=2))

    new_vocab = {}  # new empty vocab
    for word in vocab:  # for each pair in a given map
        symbols = word.split()  # ["l", "o", "w", "</w>"]
        merged = merge_pair(symbols, pair)
        new_word = " ".join(merged)
        # print(f"word={word}, new_word={new_word}")
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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Get number of merges (training cycles)
    num_merges = int(args.merges)

    # Get words from corpus (training data)
    vocab = Corpus.vocab(args.corpus)

    # Train vocab model (vocab is the set of all merges)
    merge_table = []
    for i in range(num_merges):
        # pre-process merge pairs every cycle
        pairs = get_pairs(vocab)  # create pairs
        if not pairs:  # bail if pairs is empty
            print(f"Exhausted all potential pairs! Halted at step {i}.")
            break
        # use the highest ranked pair for the next merge cycle
        best = max(pairs, key=pairs.get)  # get max rank
        merge_table.append(best)
        vocab = get_merges(vocab, best)  # merge ranked pair

    # Print vocab training results (dump merges)
    print("Merge Table:")
    print(json.dumps(merge_table, indent=2))
    print("Final Vocab:")
    print(json.dumps(vocab, indent=2))

    # Collect All Unique Tokens
    token_set = set()
    for word in vocab:  # must be the vocab!
        for symbol in word.split():
            token_set.add(symbol)

    # Assign IDs in sorted order (order matters)
    token_list = sorted(list(token_set))

    # Map each unique token (symbol) to an integer ID.
    token_to_id = {token: idx for idx, token in enumerate(token_list)}
    id_to_token = {idx: token for idx, token in enumerate(token_list)}
    tokens = [id_to_token[i] for i in sorted(id_to_token)]

    print("Tokenizer:")
    print(json.dumps(token_to_id, indent=2))

    # Build the rank table (rank merges)
    rank_table = {}
    for i, pair in enumerate(merge_table):
        token = "".join(pair)
        rank_table[token] = i

    print("Rank Table:")
    print(json.dumps(rank_table, indent=2))

    # Score the merges
    scores = {}
    for token in tokens:
        rank = rank_table.get(token)
        scores[token] = -math.log(rank + 1) if rank else -1e6

    print("Token Scores:")
    print(json.dumps(scores, indent=2))
