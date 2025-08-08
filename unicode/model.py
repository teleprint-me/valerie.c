"""
Copyright © 2025 Austin Berrio
@file unicode.model.py
@license cc-by-sa-nc-4.0
@ref https://aclanthology.org/P16-1162/
@ref https://aclanthology.org/2025.coling-main.400/
@ref https://huggingface.co/blog/catherinearnett/dangers-of-tokenizer-recycling
"""

import argparse
import collections
import json
import math

if __name__ == "__main__":
    num_merges = 15

    # Get words from corpus (training data)
    vocab = list("lo low lower newest wide wider widest x-ray")
    # print(json.dumps(vocab, indent=2))

    base_alphabet = list(dict.fromkeys(vocab))  # first-appearance order

    merge_table = []
    for i in range(num_merges):
        # pre-process pairs
        pairs = [(vocab[i], vocab[i + 1]) for i in range(len(vocab) - 1)]
        if not pairs:
            break  # Exhausted all possible occurrances
        # print(json.dumps(pairs, indent=2))

        # pair frequencies
        frequencies = {}
        for pair in pairs:
            if pair in frequencies:
                frequencies[pair] += 1
            else:
                frequencies[pair] = 1

        # pretty-print without losing structure
        printable = {f"({a!r}, {b!r})": cnt for (a, b), cnt in frequencies.items()}
        print(json.dumps(printable, indent=2, ensure_ascii=False))

        # compress duplicates (run-length encoding)
        encodings = []
        current = pairs[0]
        count = 1  # current is inclusive
        for nxt in pairs[1:]:  # skip to the next pair
            if current == nxt:
                count += 1  # pair exists, inc freqs
            else:
                encodings.append((current, count))  # track freqs
                current, count = nxt, 1  # init next pair
        encodings.append((current, count))

        # get the best pair
        best_pair = None  # ("l", "o")
        best_freq = -1  # frequency
        for pair, freq in frequencies.items():
            # break ties (mitigates collisions and overlapping boundaries)
            if (freq > best_freq) or (
                freq == best_freq and (best_pair is None or pair < best_pair)
            ):
                best_pair = pair
                best_freq = freq

        if not best_pair:
            break  # exhausted all possible merges

        print(f"best_pair={best_pair}, best_freq={best_freq}")
        merge_table.append(best_pair)

        # merge n-grams and replace all non-overlapping occurences
        n_grams = []
        i = 0
        m = 0  # count non-overlapping occurances
        a, b = best_pair  # current target pair
        while i < len(vocab):
            new_tok = None
            if i + 1 < len(vocab) and vocab[i] == a and vocab[i + 1] == b:
                new_tok = a + b
                m += 1
                i += 2
            else:
                new_tok = vocab[i]
                i += 1

            if new_tok:
                n_grams.append(new_tok)

        assert (
            len(n_grams) == len(vocab) - m
        ), f"Expected {len(vocab) - m} vocab after merge, got {len(n_grams)}"

        vocab = n_grams  # update the set of pairs

    print("Base Alphabet:")
    print(json.dumps(base_alphabet))

    print("Merge Table:")
    print(json.dumps(merge_table, indent=2))

    print("Vocab Table:")
    print(json.dumps(vocab, indent=2))

    # Assign IDs in sorted order (order matters)
    # Use base vocab to heal token set
    token_list = sorted(list(set(base_alphabet + vocab)))
    print("Tokenizer:")
    print(json.dumps(token_list, indent=2))
