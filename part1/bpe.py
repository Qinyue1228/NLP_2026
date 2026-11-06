import itertools
from collections import Counter


def pretokenize(s, begin_of_word=u"\u2581", replacement="_"):
    """This is close to the pre-tokenizer in the textbook"""
    return [("" if i == 0 else begin_of_word) + word.replace(begin_of_word, replacement) for i, word in enumerate(s.split())]

def prepare_corpus(corpus):
    return list(itertools.chain.from_iterable(pretokenize(s) for s in corpus))

def count_words(prepared_corpus):
    return Counter(tuple(word) for word in prepared_corpus)

def character_vocab(words):
    vocab = set(
        ch
        for word in words
        for ch in word
    )
    return vocab

def count_pairs(word_counts: dict):
    """
    Count adjacent symbol pairs, weighted by word frequency.

    word_counts maps tuples of symbols to their frequencies, for example:
        {("_", "l", "o", "w"): 5}
    """
    pair_counts = Counter()

    for symbols, frequency in word_counts.items():
        for left, right in zip(symbols, symbols[1:]):
            pair_counts[(left, right)] += frequency

    return pair_counts

def merge_pair_in_word(symbols, pair):
    """
    Merge all non-overlapping occurrences of pair in one tuple of symbols.

    Example:
        symbols = ("_", "l", "o", "w")
        pair = ("l", "o")
        result = ("_", "lo", "w")
    """
    merged = []
    i = 0

    while i < len(symbols):
        if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == pair:
            merged.append(symbols[i] + symbols[i + 1])
            i += 2
        else:
            merged.append(symbols[i])
            i += 1

    return tuple(merged)

def merge_pair_in_corpus(word_counts, pair):
    """Apply one merge operation to every distinct word in the corpus."""
    new_word_counts = Counter()

    for symbols, frequency in word_counts.items():
        merged_symbols = merge_pair_in_word(symbols, pair)
        new_word_counts[merged_symbols] += frequency

    return new_word_counts

def fit_bpe(prepared_corpus, max_merges, UNK=u"\u2047"):
    """
    Learn BPE merges from pretokenized corpus entries.

    Parameters
    ----------
    corpus : iterable of str
        Entries such as ["_low", "_lower", "_newest"].
        Use prepare_corpus(...) when starting from ordinary text.
    max_merges : int
        Maximum number of BPE merge operations to learn.
        The vocabulary size is the number of characters seen _plus_ the number of merges (up to this maximum).
        

    Returns
    -------
    dict with:
        "merges"          : learned pairs, in learning order
        "vocabulary"      : all character and merged symbols, plus UNK
        "character_vocab" : characters observed in training
    """
    # Each corpus entry begins as a tuple of individual characters.
    word_counts = count_words(prepared_corpus)

    ch_vocab = character_vocab(word_counts.keys())

    vocabulary = set(ch_vocab)
    vocabulary.add(UNK)

    merges = []

    for n in range(max_merges):
    
        pair_counts = count_pairs(word_counts)
        if not pair_counts: # nothing left to merge
            break

        # Highest frequency wins. For ties, choose alphabetically
        best_pair = min(
            pair_counts,
            key=lambda pair: (-pair_counts[pair], pair)
        )

        merges.append(best_pair)
        

        # The merged unit becomes a vocabulary item, e.g. ("l", "o") -> "lo".
        vocabulary.add(best_pair[0] + best_pair[1])

        word_counts = merge_pair_in_corpus(word_counts, best_pair)

    return {
        "merges": merges,
        "vocabulary": vocabulary,
        "character_vocab": ch_vocab,
        "UNK": UNK
    }

def segment_entry(entry, model):
    """
    Segment one pretokenized entry using a fitted BPE model.

    Unknown characters are replaced by the atomic <UNK> symbol.

    Example:
        entry = "_lowest"
        output might be ["_low", "e", "s", "t"]
    """
    ch_vocab = model["character_vocab"]

    # Unknown characters are replaced before applying BPE merges.
    symbols = tuple(
        character if character in ch_vocab else model["UNK"]
        for character in entry
    )

    # Apply learned merges in the order in which they were learned.
    for pair in model["merges"]:
        symbols = merge_pair_in_word(symbols, pair)

    return list(symbols)

def segment_text(text, model):
    """
    Pretokenize and segment a raw text.

    Returns one list of BPE units per original word.
    """
    entries = pretokenize(text)
    return list(itertools.chain.from_iterable(segment_entry(entry, model) for entry in entries))

