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


class Model:
    """Byte-pair Encoding"""

    @staticmethod
    def pairs(vocab: dict[str, int]) -> dict[tuple[str, str], int]:
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
        # print("Updated pairs:")
        # print(json.dumps(vocab, indent=2))

        new_vocab = {}  # new empty vocab
        for word in vocab:  # for each pair in a given map
            symbols = word.split()  # ["l", "o", "w", "</w>"]
            bigram = Model.bigram(symbols, pair)  # merge neighbors
            new_word = " ".join(bigram)  # new n-gram
            # print(f"word={word}, new_word={new_word}")
            new_vocab[new_word] = vocab[word]
        return new_vocab


class Tokenizer:
    def __init__(self, vocab: dict[str, int]):
        self.vocab = vocab
        self.merges = []

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

    @property
    def tokens(self) -> list[str]:
        # Collect All Unique Tokens
        token_set = set()
        for word in self.vocab:  # must be the vocab!
            for symbol in word.split():
                token_set.add(symbol)
        # Assign IDs in sorted order (order matters)
        return sorted(list(token_set))

    @property
    def token_to_id(self) -> dict[str, int]:
        return {token: idx for idx, token in enumerate(self.tokens)}

    @property
    def id_to_token(self, id: int) -> str:
        return {idx: token for idx, token in enumerate(self.tokens)}

    @property
    def ranks(self) -> dict[str, int]:
        # Build the rank table (rank merges)
        rank_table = {}
        for i, pair in enumerate(self.merges):
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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Get number of merges (training cycles)
    num_merges = int(args.merges)

    # Get words from corpus (training data)
    vocab = Corpus.vocab(args.corpus)

    # Train vocab model (vocab is the set of all merges)
    tokenizer = Tokenizer(vocab)
    tokenizer.train(args.merges)

    # Print vocab training results (dump merges)
    print("Merge Table:")
    print(json.dumps(tokenizer.merges, indent=2))

    print("Final Vocab:")
    print(json.dumps(tokenizer.vocab, indent=2))

    print("Tokenizer:")
    print(json.dumps(tokenizer.token_to_id, indent=2))

    # Build the rank table (rank merges)
    print("Rank Table:")
    print(json.dumps(tokenizer.ranks, indent=2))

    # Score the merges
    print("Token Scores:")
    print(json.dumps(tokenizer.scores, indent=2))
