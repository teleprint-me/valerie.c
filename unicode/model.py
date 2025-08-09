"""
Copyright © 2025 Austin Berrio
@file unicode.model.py
@license cc-by-sa-nc-4.0
@ref https://arxiv.org/abs/1508.07909
@ref https://arxiv.org/abs/2409.11501
@ref https://arxiv.org/abs/2507.07955
@ref https://huggingface.co/blog/catherinearnett/dangers-of-tokenizer-recycling
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
        return vocab


class Model:
    """Byte-pair Encoding"""

    @staticmethod
    def pairs(vocab: dict[str, int]) -> dict[tuple[str, str], int]:
        pairs = collections.defaultdict(int)  # init freqs to 0
        for word, freq in vocab.items():  # unpacks ("l o w", 5)
            symbols = word.split()  # split word by char -> ["l", "o", "w", ...]
            for i in range(len(symbols) - 1):  # for each step in the set of symbols
                cur = symbols[i]  # "l"
                nxt = symbols[i + 1]  # "o"
                pairs[cur, nxt] += freq  # p[("l", "o")] += 1
        return pairs  # {('l', 'o'): 1}

    @staticmethod
    def bigram(symbols: list[str], pair: tuple[str, str]) -> list[str]:
        bigram = []
        i = 0
        while i < len(symbols):
            # If this symbol and the next match the pair, merge them
            if (
                i < len(symbols) - 1
                and symbols[i] == pair[0]
                and symbols[i + 1] == pair[1]
            ):
                bigram.append(symbols[i] + symbols[i + 1])
                i += 2  # Skip the next symbol (it's merged)
            else:
                bigram.append(symbols[i])
                i += 1
        return bigram

    @staticmethod
    def merges(vocab: dict[str, int], pair: tuple[str, str]) -> dict[str, int]:
        new_vocab = {}  # new empty vocab
        for word in vocab:  # for each pair in a given map
            symbols = word.split()  # ["l", "o", "w", ...]
            bigram = Model.bigram(symbols, pair)  # merge neighbors
            new_word = " ".join(bigram)  # new n-gram
            new_vocab[new_word] = vocab[word]
        return new_vocab


class Tokenizer:
    def __init__(self, vocab: dict[str, int]):
        self.model = {
            "type": "BPE",
            "version": "0.1.0",
            "vocab": vocab,
            "merges": [],
        }

    def __len__(self) -> int:
        return len(self.token_to_id)

    @property
    def type(self) -> str:
        return self.model["type"]

    @property
    def version(self) -> str:
        return self.model["version"]

    @property
    def vocab(self) -> dict[str, int]:
        return self.model["vocab"]

    @vocab.setter
    def vocab(self, value: dict[str, int]) -> None:
        self.model["vocab"] = value

    @property
    def merges(self) -> list[tuple[str, str]]:
        return self.model["merges"]

    @merges.setter
    def merges(self, value: list[tuple[str, str]]):
        self.model["merges"] = value

    @property
    def tokens(self) -> list[str]:
        # Collect All Unique Tokens
        token_set = set()
        for word in self.vocab:  # must be vocab!
            for symbol in word.split():
                token_set.add(symbol)
        # Assign IDs in sorted order (order matters)
        return sorted(list(token_set))

    @property
    def token_to_id(self) -> dict[str, int]:
        return {token: idx for idx, token in enumerate(self.tokens)}

    @property
    def id_to_token(self) -> dict[int, str]:
        return {idx: token for idx, token in enumerate(self.tokens)}

    @property
    def ranks(self) -> dict[str, int]:
        # Build the rank table (rank merges)
        rank_table = {}
        for i, pair in enumerate(self.merges):  # must be merges!
            token = "".join(pair)
            rank_table[token] = i
        return rank_table

    @property
    def scores(self):
        # Score the merges
        scores = {}
        for token in self.tokens:
            rank = self.ranks.get(token)
            scores[token] = -math.log(rank + 1) if rank else -1e6
        return scores

    def train(self, num_merges: int) -> None:
        # Train vocab model (vocab is the set of all merges)
        self.merges = []
        for i in range(num_merges):
            # pre-process merge pairs every cycle
            pairs = Model.pairs(self.vocab)  # create pairs
            if not pairs:  # bail if pairs is empty
                print(f"Exhausted all potential pairs! Halted at step {i}.")
                break
            # use the highest ranked pair for the next merge cycle
            best = max(pairs, key=pairs.get)  # get max rank
            self.merges.append(best)
            self.vocab = Model.merges(self.vocab, best)  # merge ranked pair

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as file:
            json.dump(self.model, file, ensure_ascii=False, indent=2)

    def load(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as file:
            self.model = json.load(file)

    def encode(self, token: str) -> int:
        return self.token_to_id[token]

    def decode(self, id: int) -> str:
        return self.id_to_token[id]


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
        "-l",
        "--load",
        required=False,
        type=str,
        default=None,
        help="load model from file",
    )
    parser.add_argument(
        "-s",
        "--save",
        required=False,
        type=str,
        default=None,
        help="save model to file",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        required=False,
        type=bool,
        default=False,
        help="enable debug mode",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Get number of merges (training cycles)
    num_merges = int(args.merges)

    # Get words from corpus (training data)
    vocab = Corpus.vocab(args.corpus)

    # Train vocab model (vocab is the set of all merges)
    tokenizer = Tokenizer(vocab)

    if args.load and args.save:
        raise Exception("Use save or load. Not both.")

    if args.load and not args.save:
        tokenizer.load(args.load)

    if args.save and not args.load:
        tokenizer.train(args.merges)
        tokenizer.save(args.save)

    if args.verbose:
        # Print vocab training results (dump merges)
        print("Model:")
        print(json.dumps(tokenizer.model, indent=2))

        # Build the rank table (rank merges)
        print("Rank Table:")
        print(json.dumps(tokenizer.ranks, indent=2))

        # Score the merges
        print("Token Scores:")
        print(json.dumps(tokenizer.scores, indent=2))

    print("Tokenizer:")
    print(json.dumps(tokenizer.token_to_id, indent=2))
    print(f"Model has {len(tokenizer)} tokens.")
