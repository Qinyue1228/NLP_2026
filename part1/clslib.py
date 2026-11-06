import scipy
import numpy as np
from scipy.special import softmax
from collections import OrderedDict
from collections import Counter
from tokenizers import Tokenizer


class NamedClasses:
    """
    This is a helper class that wraps around a list of class names and provides 
    functions to map from names (str) to integers.
    """

    def __init__(self, names: list):
        self.names = tuple(names)        
        self._str2int = {name: k for k, name in enumerate(names)}
    
    def index(self, name: str):
        """Convert name to 0-based integer (0-based because Python lists are 0-based)"""
        return self._str2int[name]  # if the name is not recognised, this will raise a ValueError

    def name(self, index: int):
        """Convert an 0-based integer to a class name"""
        return self.names[index]  # if the index is out of bounds, this will raise a IndexError        

    def __len__(self):
        """Size of label space"""
        return len(self.names)

    def __iter__(self):
        """Iterate over names"""
        return iter(self.names)

    def items(self):
        """Iterates over pairs of the kind (index, name) for all categories in the space"""
        return enumerate(self.names)

    def __str__(self):
        return ", ".join(f"{i}: {n}" for i, n in self.items())

    def __repr__(self):
        return f"{type(self).__name__}({repr(self.names)})"
        
class ProbabilisticModel:
    """
    A probability model is able to map input text x to a probability mass function. 
    """

    def __call__(self, x: str) -> 'np.array':
        raise NotImplementedError("Implement me in a sub-class")

class TabularModel(ProbabilisticModel):
    
    @staticmethod
    def infer_num_classes(pmfs: dict) -> int:
        """
        Inspects the conditional pmfs, dicts mapping conditions to K-dimensional probability vectors, to infer K.
        """
        K = None
        for x, probs in pmfs.items():
            if K is None:
                K = len(probs)
            else:
                if K != len(probs):
                    raise ValueError(f"Every pmf must be specified by a {K}-dimensional vector, got {len(probs)} for {x}")
        return K
        
    def __init__(self, pmfs: dict):
        """
        """
        self.num_classes = TabularModel.infer_num_classes(pmfs)        
        self.pmfs = pmfs

    def __call__(self, x: str) -> 'np.array':
        pmf = self.pmfs.get(x, None)
        if pmf is None:
            raise ValueError(f"I do not know a pmf for: {x}")
        return pmf

class DecisionRule:
    """
    This class outputs a decision by searching through a given pmf via the method __call__.
    """

    def __call__(self, pmf: 'np.array') -> int:
        """Return the chosen outcome (int) out of those available in the support of the pmf"""
        raise NotImplementedError("Implement me in a sub-class")        

class MostProbableClass(DecisionRule):
    """
    The 'most probable class' decision rule.
    """
    
    def __call__(self, pmf: 'np.array') -> int:
        return np.argmax(pmf)

class TextClassifier:
    """
    A text classifier is able to map text (str) to a class (str), 
    we do so via the special method __call__.
    """

    def __init__(self, outcome_space: NamedClasses, model: ProbabilisticModel, decoder: DecisionRule):
        self.outcome_space = outcome_space
        self.model = model
        self.decoder = decoder
    
    def __call__(self, x: str) -> str:
        """Map input text x to a named category (str)"""
        pmf = self.model(x)
        index = self.decoder(pmf)
        return self.outcome_space.name(index)        

class Encoder:

    def __len__(self) -> int:
        """Dimensionality of the representation space"""
        raise NotImplementedError("Concrete implementations will be discussed later")

    def __call__(self, x: str) -> 'np.array':
        """Represent x as a D-dimensional vector of real numbers"""
        raise NotImplementedError("Implement me in a sub-class")

class ToyEncoder(Encoder):

    def __init__(self, vocab: list):
        """
        This toy encoder has one dimension for each symbol in 'vocab'. 
        These known 'words' are detected by exact string matching against the tokens returned by string.split().
        """
        self._vocab = tuple(vocab)
    
    def __len__(self):
        """Number of dimensions"""
        return len(self._vocab) 

    def coords(self):
        """The mnemonic labels for the axes are the words in the vocabulary"""
        return [f"'{w}' in x" for w in self._vocab]

    def __call__(self, x):
        """
        Our toy encoder employs a crude tokeniser (ie, python's string.split method) 
        and detects the presence in x of the words in its vocabulary.
        Detection is based on string matching, after lowercasing.
        """
        tokens = [t.lower() for t in x.split()]
        return np.array([1 if w in tokens else 0 for w in self._vocab])        

class LogLinearSoftmaxRegressor(ProbabilisticModel):

    def __init__(self, encoder: Encoder, W: 'np.array', b: 'np.array'):
        """
        encoder: turns a text (str) into a D-dimensional vector
        W: KxD matrix of weights
        b: B-dimensional vector of weights
        """
        self.encoder = encoder
        assert W.shape[0] == len(b), f"The matrix W should have shape KxD with K={len(b)}"
        assert W.shape[1] == len(encoder), f"The matrix W should have shape KxD with D={len(encoder)}"
        self.W = W
        self.b = b
        
    def __call__(self, x: str) -> 'np.array':
        h = self.encoder(x) # D features
        s = np.dot(self.W, h) + self.b # K linear predictors
        pmf = softmax(s) # K probabilities
        return pmf


class TokenIndicator(Encoder):

    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer
        
    def __len__(self) -> int:
        """Dimensionality of the representation space"""
        return len(self.tokenizer.get_vocab())

    def __call__(self, x: str) -> 'np.array':
        """Represent x as a D-dimensional vector of real numbers"""
        unique_token_ids = list(set(self.tokenizer.encode(x).ids))

        # A 0 vector with the same length as the vocabulary except in positions corresponding 
        # to tokens appearing in the document
        vector = np.zeros(len(self))
        vector[unique_token_ids] = 1
        return vector


class MajorityVoteClassifier:
    """
    A text classifier is able to map text (str) to a class (str), 
    we do so via the special method __call__.
    """    

    def __init__(self, outcome_space: NamedClasses, models: list[ProbabilisticModel], decoder: DecisionRule, seed=0):
        """
        outcome_space: NamedClasses for the output space
        models: a list of trained instances of ProbabilisticModel
        decoder: a DecisionRule
        seed: we break ties at random
        """
        self.outcome_space = outcome_space
        self.models = models
        self.decoder = decoder
        self.rng = np.random.RandomState(seed)
    
    def __call__(self, x: str) -> str:
        """Map input text x to a named category (str)"""        
        # compute M decisions
        candidates = [self.decoder(model(x)) for model in self.models]
        # counte them
        values, counts = np.unique(candidates, return_counts=True)        
        # break ties at random
        index = self.rng.choice(values[counts == counts.max()])
        # category's name
        return self.outcome_space.name(index)        