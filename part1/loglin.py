from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression
import pickle
from tqdm.auto import tqdm

def load(path):
    """
    Load a logistic model from {path}.pickle and return it.
    """
    with open(f"{path}.pickle", 'rb') as f:
        logistic_model = pickle.load(f)
    return logistic_model

def fit(training_set, encoder: 'Encoder | None', save_to=None, max_iter=100, n_jobs=-1, **encoder_args):
    """
    Constructs, trains and returns a softmax regressor.
    
    training_set: list of text-label pairs (a text is a string, a label is a 0-based integer)
    encoder: an instance of the Encoder class (such as TokenIndicator)    
    save_to: (str) if given, save the model to {save_to}.pickle
    """
    # Convert the training and validation sets to vectors
    if encoder is None:
        try:
            from .experiment import train_tokenizer
        except ImportError:
            from experiment import train_tokenizer
        encoderdict = {}
        encoderdict["vocab_size"] = encoder_args.get("vocab_size", 1000)
        encoderdict["min_frequency"] = encoder_args.get("min_frequency", 2)
        EncoderClass = encoder_args["encoderclass"]
        encoder = EncoderClass(
            train_tokenizer(
                [text for text, label in training_set],
                **encoderdict,
            )
        )

    training_vectors = [encoder(text) for text, label in training_set]
    training_labels = [label for text, label in training_set]

    #To train the model more efficiently, we transform the training vectors to a sparse matrix
    X_train_sparse = csr_matrix(training_vectors)

    # Train the logistic model
    logistic_model = LogisticRegression(solver="saga", max_iter=max_iter, n_jobs=n_jobs) #sparse matrix frinedly solver 
    logistic_model.fit(X_train_sparse, training_labels)

    if save_to is not None:        
        with open(f"{save_to}.pickle", 'wb') as f:
            pickle.dump(logistic_model, f, protocol=pickle.HIGHEST_PROTOCOL)
        
    return logistic_model        

def predict(texts, model, encoder: 'Encoder'):
    """
    Predicts labels (via the most probable class decision rule) for a collection of texts. 
    (scikit-learn hard-codes the most probable class as the decision rule of choice)
    
    texts: a list of texts, each a string
    model: a trained instance of sklearn's LogisticRegression
    encoder: the Encoder used for in the model's training
    """
    validation_vectors = [encoder(text) for text in texts]
    X_val_sparse = csr_matrix(validation_vectors)
    predictions = model.predict(X_val_sparse)
    return predictions

def predict_probs(texts, model, encoder: 'Encoder'):
    """
    Predicts labels (via the most probable class decision rule) for a collection of texts. 
    (scikit-learn hard-codes the most probable class as the decision rule of choice)
    
    texts: a list of texts, each a string
    model: a trained instance of sklearn's LogisticRegression
    encoder: the Encoder used for in the model's training
    """
    validation_vectors = [encoder(text) for text in texts]
    X_val_sparse = csr_matrix(validation_vectors)
    probs = model.predict_proba(X_val_sparse)
    return probs