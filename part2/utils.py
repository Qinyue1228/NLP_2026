from collections import OrderedDict
from tabulate import tabulate
import string
import numpy as np
import nltk
import random


def boc_emb(word: str, vocab=string.ascii_lowercase) -> 'np.array':
    """
    Produce a bag-of-characters embedding for a given word, in the space whose coordinates stand for
        0: count of characters not in string.ascii_lowercase
        i (other than 0): count of the ith character in string.ascii_lowercase
    """
    c2i = {c: i for i, c in enumerate(vocab, 1)}
    dim = len(c2i) + 1 # let's use 0 for everything not in vocab
    vector = np.zeros(dim)
    for c in word.lower():
        vector[c2i.get(c, 0)] += 1
    return vector

def embeddings_to_str(embeddings: dict, floatfmt="2.2f") -> str:
    """
    Render an embedding table as a table-formatted string.
    
    embeddings (dict): a key is a known word, its value is a vector (or np.array) of coordinate values (each a float)    
    """
    table = []
    num_dims = None
    for w, vec in embeddings.items():
        table.append([w] + list(vec))
        if num_dims is None:
            num_dims = len(vec)
        elif num_dims != len(vec):
            raise ValueError(f"All vectors should have the same dimensionality; I got {len(vec)} for {w}, but before that it was {num_dims}.")  
    return tabulate(table, floatfmt=floatfmt, tablefmt="plain")

def sample_pos_and_neg_words(emb_model: 'PretrainedEmbeddings', size=1000, seed=42):        
    try: 
        from nltk.corpus import opinion_lexicon
    except ImportError:
        nltk.download("opinion_lexicon")       
        from nltk.corpus import opinion_lexicon
    
    rng = random.Random(seed)
    # Get lists of positive and negative words
    pos_words = [w for w in opinion_lexicon.positive() if w in emb_model]
    neg_words = [w for w in opinion_lexicon.negative() if w in emb_model]
    # Remove overlap
    pos_words = list(set(pos_words) - set(neg_words))
    neg_words = list(set(neg_words) - set(pos_words))
    # Shuffle
    rng.shuffle(pos_words)
    rng.shuffle(neg_words)
    # Subset
    return pos_words[:size], neg_words[:size]        


