
from scipy.sparse import csr_matrix
from torch.utils.data import Dataset
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm.auto import tqdm
import time
import scipy
import scipy.sparse
import json
import numpy as np
from sklearn.model_selection import GridSearchCV

class SparseMLP(nn.Module):
    """One-hidden-layer MLP that accepts sparse input tensors.

    Inputs: feature vectors of size `input_dim` (e.g. bag-of-words).
    Process: sparse matmul into a dense hidden layer, ReLU + dropout, linear (num_classes) and log-softmax outputs
        log-softmax, rather than softmax, for numerical stability
    Output: logprobs over `num_classes`.
    """

    def __init__(self, input_dim, hidden_dim, num_classes, dropout=0.2):
        """
        input_dim: size of the input feature space (e.g. vocabulary size)
        hidden_dim: width of the hidden layer
        num_classes: number of output labels
        dropout: fraction of hidden units zeroed during training (regularisation)
        """
        super().__init__()

        # First-layer weights stay dense; inputs arriving here are sparse.
        self.weight = nn.Parameter(
            torch.empty(input_dim, hidden_dim)
        )
        self.bias = nn.Parameter(
            torch.zeros(hidden_dim)
        )

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # Everything after the first layer is dense.
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, num_classes),            
            nn.LogSoftmax(dim=-1) # for numerical stability, it is common to use log softmax, rather than softmax
        )

        nn.init.xavier_uniform_(self.weight) # why xavier?

    def forward(self, x):
        """Map a sparse batch [batch, input_dim] to class log-probabilities [batch, num_classes]."""
        # First matmul keeps x sparse; result is dense from here on.
        x = torch.sparse.mm(x, self.weight)
        x = x + self.bias

        x = self.activation(x)
        x = self.dropout(x)
        # logprobs
        return self.output(x)


class SparseMMLP(nn.Module):
    """Two-hidden-layer MLP that accepts sparse input tensors.

    Inputs: feature vectors of size `input_dim` (e.g. bag-of-words).
    Process: sparse matmul into a dense hidden layer, ReLU + dropout, linear (num_classes) and log-softmax outputs
        log-softmax, rather than softmax, for numerical stability
    Output: logprobs over `num_classes`.
    """

    def __init__(self, input_dim, hidden_dim, num_classes, dropout=0.2):
        """
        input_dim: size of the input feature space (e.g. vocabulary size)
        hidden_dim: width of the hidden layer
        num_classes: number of output labels
        dropout: fraction of hidden units zeroed during training (regularisation)
        """
        super().__init__()

        # First-layer weights stay dense; inputs arriving here are sparse.
        self.weight = nn.Parameter(
            torch.empty(input_dim, hidden_dim)
        )
        self.bias = nn.Parameter(
            torch.zeros(hidden_dim)
        )

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # Everything after the first layer is dense.
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(64, num_classes),            
            nn.LogSoftmax(dim=-1) # for numerical stability, it is common to use log softmax, rather than softmax
        )

        nn.init.xavier_uniform_(self.weight) # why xavier?

    def forward(self, x):
        """Map a sparse batch [batch, input_dim] to class log-probabilities [batch, num_classes]."""
        # First matmul keeps x sparse; result is dense from here on.
        x = torch.sparse.mm(x, self.weight)
        x = x + self.bias

        x = self.activation(x)
        x = self.dropout(x)
        # logprobs
        return self.output(x)

def scipy_to_torch_sparse(x):
    """Convert a SciPy sparse matrix to a coalesced PyTorch sparse COO tensor.

    Input: SciPy sparse matrix.
    Output: torch sparse tensor with the same shape and non-zero values.
    """
    x = x.tocoo()
    indices = torch.tensor([x.row, x.col], dtype=torch.long)
    values = torch.tensor(x.data, dtype=torch.float32)

    return torch.sparse_coo_tensor(indices, values, size=x.shape).coalesce()

class SparseDataset(Dataset):
    """Wraps a SciPy sparse feature matrix and label list for use with DataLoader.

    Inputs: X (sparse matrix of examples), y (integer labels).
    Output per item: one sparse row and its label tensor.
    """

    def __init__(self, X, y):
        self.X = X
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def sparse_collate(batch):
    """Collate individual sparse rows into one PyTorch sparse batch.

    Input: list of (sparse row, label) pairs from SparseDataset.
    Process: stack SciPy rows, convert to a torch sparse tensor, stack labels.
    Output: (X_batch sparse tensor, y_batch long tensor).
    """
    X_rows, y_rows = zip(*batch)

    # Stack SciPy sparse rows, then convert the whole batch at once.
    X_batch = scipy.sparse.vstack(X_rows)
    X_batch = scipy_to_torch_sparse(X_batch)

    y_batch = torch.stack(y_rows)

    return X_batch, y_batch


def train_one_epoch(model, dataloader, optimizer, criterion, device="cpu"):
    """Run one training epoch: forward, loss, backward, optimizer step.

    Inputs: model, train dataloader, optimizer, loss function, device.
    Output: dict with average train loss and accuracy for the epoch.
    """
    if isinstance(device, str):
        device = torch.device(device)
    
    model.train()  # enable dropout / training behaviour

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for X_batch, y_batch in dataloader:
        # move the batch to the device
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()  # clear gradients from the previous step

        logprobs = model(X_batch) # forward pass
        loss = criterion(logprobs, y_batch) # compute the loss

        loss.backward()   # compute gradients
        optimizer.step()  # update parameters

        # accumulate weighted loss and correct predictions for epoch averages
        total_loss += loss.item() * y_batch.size(0)
        predictions = logprobs.argmax(dim=1) # get the predicted class
        total_correct += (predictions == y_batch).sum().item() # count the number of correct predictions
        total_examples += y_batch.size(0) # count the number of examples

    return {
        "loss": total_loss / total_examples, # average loss
        "accuracy": total_correct / total_examples, # average accuracy
    }

def evaluate(model, dataloader, criterion, device="cpu"):
    """Evaluate the model without updating weights.

    Inputs: model, eval dataloader, loss function, device.
    Output: (metrics dict with loss/accuracy, numpy array of predicted labels).
    """
    if isinstance(device, str):
        device = torch.device(device)
        
    model.eval()  # disable dropout / training behaviour

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    all_preds = []
    with torch.no_grad():  # no gradients needed for evaluation

        for X_batch, y_batch in dataloader:
            # move the batch to the device
            X_batch = X_batch.to(device) 
            y_batch = y_batch.to(device)

            logprobs = model(X_batch) # forward pass

            loss = criterion(logprobs, y_batch) # compute the loss

            # accumulate metrics; also keep predictions for later use
            total_loss += loss.item() * y_batch.size(0)
            predictions = logprobs.argmax(dim=1) # get the predicted class
            all_preds.append(predictions)

            total_correct += (predictions == y_batch).sum().item() # count the number of correct predictions
            total_examples += y_batch.size(0)

    # concatenate batch predictions into one array
    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples
    }, torch.cat([t.detach().cpu() for t in all_preds]).numpy()


def _evaluate_inner_candidate(mmlp: bool, candidate, training_set, X_train_sparse, labels,
                              fold_splits, num_classes, input_dim, encoder,
                              encoder_class, encoder_args):
    """Evaluate one inner-CV candidate in a CPU-only worker process."""
    torch.set_num_threads(1)
    from sklearn.model_selection import ParameterGrid

    parameters = candidate
    fold_losses = []
    cpu_device = torch.device("cpu")

    for train_indices, fold_indices in fold_splits:
        if encoder is None:
            try:
                from .experiment import train_tokenizer
            except ImportError:
                from experiment import train_tokenizer
            fold_training_set = [training_set[index] for index in train_indices]
            fold_validation_set = [training_set[index] for index in fold_indices]
            fold_training_text = [text for text, label in fold_training_set]
            fold_encoder = encoder_class(
                tokenizer = train_tokenizer(
                    fold_training_text,
                    **encoder_args,
                ),
                corpus = fold_training_text
            )
            fold_X_train_sparse = csr_matrix([
                fold_encoder(text) for text, label in fold_training_set
            ])
            fold_X_validation_sparse = csr_matrix([
                fold_encoder(text) for text, label in fold_validation_set
            ])
            fold_input_dim = len(fold_encoder)
        else:
            fold_X_train_sparse = X_train_sparse[train_indices]
            fold_X_validation_sparse = X_train_sparse[fold_indices]
            fold_input_dim = input_dim

        fold_train_dataset = SparseDataset(fold_X_train_sparse, labels[train_indices])
        fold_validation_dataset = SparseDataset(
            fold_X_validation_sparse,
            labels[fold_indices],
        )
        fold_train_loader = DataLoader(
            fold_train_dataset,
            batch_size=parameters["batch_size"],
            shuffle=True,
            collate_fn=sparse_collate,
        )
        fold_validation_loader = DataLoader(
            fold_validation_dataset,
            batch_size=parameters["batch_size"],
            shuffle=False,
            collate_fn=sparse_collate,
        )
        model = SparseMMLP(
            input_dim=fold_input_dim,
            hidden_dim=parameters["hidden_dim"],
            num_classes=num_classes,
            dropout=parameters["dropout"],
        ).to(cpu_device) if mmlp else SparseMLP(
            input_dim=fold_input_dim,
            hidden_dim=parameters["hidden_dim"],
            num_classes=num_classes,
            dropout=parameters["dropout"],
        ).to(cpu_device)
        criterion = nn.NLLLoss()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=parameters["learning_rate"],
            weight_decay=parameters["weight_decay"],
        )

        for epoch in range(parameters["num_epochs"]):
        #     train_one_epoch(model, fold_train_loader, optimizer, criterion, cpu_device)
        # fold_metrics, _ = evaluate(
        #     model,
        #     fold_validation_loader,
        #     criterion,
        #     cpu_device,
        # )
        # fold_losses.append(fold_metrics["loss"])
            # Train the model for one epoch
            train_metrics = train_one_epoch(model, fold_train_loader, optimizer, criterion, cpu_device)
            # Evaluate the model on the validation set
            val_metrics, preds = evaluate(
                model,
                fold_validation_loader,
                criterion,
                cpu_device,
            )
            
            # Saving the best model according to validation loss
            if epoch != 0:
                if val_metrics["loss"] <= best_val_loss:
                    best_val_loss = val_metrics["loss"]
            else:
                best_val_loss = val_metrics["loss"]


    return parameters, best_val_loss

def train_neural_network_model(training_set, validation_set, num_classes, encoder: 'Encoder | None', save_to,   
                               mmlp=False,
                                hidden_dim=128, dropout=0.1, batch_size=64, learning_rate=1e-3, 
                                weight_decay=1e-4, num_epochs=20, device=torch.device("cpu"), **encoder_args):
    """Encode texts, train SparseMLP, and checkpoint the best validation model.

    Inputs: train/val (text, label) pairs, num_classes, encoder (text -> sparse vector), path prefix to save weights,
            plus standard training hyperparameters.
    Process: vectorise documents, build sparse dataloaders, train for `num_epochs`, save best-by-val-loss.
    Output: per-epoch metrics list, best epoch index, and validation predictions from the best model.
    """
    if isinstance(device, str):
        device = torch.device(device)

    if encoder is not None and not callable(encoder):
        try:
            from .clslib import TokenIndicator
        except ImportError:
            from clslib import TokenIndicator
        encoder = TokenIndicator(encoder)

    if encoder is None:
        try:
            from .experiment import train_tokenizer
            from .clslib import TokenIndicator
        except ImportError:
            from experiment import train_tokenizer
            from clslib import TokenIndicator
        encoderdict = {}
        encoderdict["vocab_size"] = encoder_args.get("vocab_size", 1000)
        encoderdict["min_frequency"] = encoder_args.get("min_frequency", 2)
        EncoderClass = encoder_args["encoderclass"]
        training_text = [text for text, label in training_set]
        encoder = EncoderClass(
            tokenizer = train_tokenizer(
                training_text,
                **encoderdict,
            ),
            corpus = training_text
        )
        
    time_start = time.time()
    # Convert the training and validation sets to vectors
    print("Transforming documents...")
    training_vectors = [encoder(text) for text, label in training_set]
    training_labels = [int(label) for text, label in training_set]
    validation_vectors = [encoder(text) for text, label in validation_set]
    validation_labels = [int(label) for text, label in validation_set]

    time_document_to_vector = time.time()
    print(f"Document transformation done! Time taken: {time_document_to_vector - time_start:.3f} seconds")
    
    # Keep SciPy sparse here; conversion to torch happens in sparse_collate
    X_train_sparse = csr_matrix(training_vectors)
    X_val_sparse = csr_matrix(validation_vectors)

    train_dataset = SparseDataset(X_train_sparse, training_labels) # create the training dataset
    val_dataset = SparseDataset(X_val_sparse, validation_labels) # create the validation dataset

    # create the data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=sparse_collate,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=sparse_collate,
    )

    time_dataloader = time.time()
    print(f"Dataloader done! Time taken: {time_dataloader - time_document_to_vector:.3f} seconds")

    # save architecture-defining arguments
    with open(f"{save_to}.json", "w", encoding="utf-8") as f:
        args = {
            'input_dim': len(encoder), 
            'hidden_dim': hidden_dim, 
            'num_classes': num_classes, 
            'dropout': dropout,
        }
        json.dump(args, f, indent=2, ensure_ascii=False)
        
    # Create the model based on the input dimension, chosen hidden dimension, number of classification classes and dropout rate
    model = SparseMMLP(input_dim=len(encoder), hidden_dim=hidden_dim, num_classes=num_classes, dropout=dropout).to(device) if mmlp else SparseMLP(input_dim=len(encoder), hidden_dim=hidden_dim, num_classes=num_classes, dropout=dropout).to(device)
    criterion = nn.NLLLoss() # create the loss function
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay) # create the optimizer

    metrics = []
    print("Training...")
    predictions = None
    for epoch in tqdm(range(num_epochs)):
        # Train the model for one epoch
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device)
        # Evaluate the model on the validation set
        val_metrics, preds = evaluate(model, val_loader, criterion, device)
        # Store the metrics for this epoch
        metrics.append({"train_metrics": train_metrics, "val_metrics": val_metrics})

        print(
            f"Epoch {epoch + 1:02d} | "
            f"train loss: {train_metrics['loss']:.4f} | "
            f"train acc: {train_metrics['accuracy']:.4f} | "
            f"val loss: {val_metrics['loss']:.4f} | "
            f"val acc: {val_metrics['accuracy']:.4f}"
            )
        
        # Saving the best model according to validation loss
        if epoch != 0:
            if val_metrics["loss"] <= best_val_loss:
                best_val_loss = val_metrics["loss"]
                torch.save(model.state_dict(), f"{save_to}.pt")
                best_epoch = epoch
                predictions = preds
        else:
            best_val_loss = val_metrics["loss"]
            torch.save(model.state_dict(), f"{save_to}.pt")
            best_epoch = epoch
            predictions = preds
    
    time_training = time.time()
    print(f"Training done! Time taken: {time_training - time_dataloader:.3f} seconds")

    return metrics, best_epoch, predictions


def train_neural_network_model_cv(training_set, validation_set, num_classes, encoder: 'Encoder | None', save_to, param_grid,  
                                  mmlp = False,
                                hidden_dim=128, dropout=0.1, batch_size=64, learning_rate=1e-3, 
                                weight_decay=1e-4, num_epochs=20, device=torch.device("cpu"),
                                inner_cv=5, random_state=0, n_jobs=-1, **encoder_args):
    """Tune SparseMLP hyperparameters with an inner stratified k-fold search.

    The outer ``validation_set`` is used only after hyperparameters have been
    selected. The selected configuration is then retrained on all of
    ``training_set`` and evaluated on that outer validation set.
    """
    if isinstance(device, str):
        device = torch.device(device)

    retrain_encoder = (encoder is None)

    from sklearn.model_selection import ParameterGrid, StratifiedKFold

    time_start = time.time()
    print("Transforming documents...")
    if encoder is not None:
        training_vectors = [encoder(text) for text, label in training_set]
        validation_vectors = [encoder(text) for text, label in validation_set]
        X_train_sparse = csr_matrix(training_vectors)
        X_val_sparse = csr_matrix(validation_vectors)
    else:
        try:
            from .experiment import train_tokenizer
        except ImportError:
            from experiment import train_tokenizer
        encoderdict = {}
        encoderdict["vocab_size"] = encoder_args.get("vocab_size", 1000)
        encoderdict["min_frequency"] = encoder_args.get("min_frequency", 2)
        # print(encoder_args)
        EncoderClass = encoder_args["encoderclass"]

    training_labels = [int(label) for text, label in training_set]
    validation_labels = [int(label) for text, label in validation_set]

    time_document_to_vector = time.time()
    print(f"Document transformation done! Time taken: {time_document_to_vector - time_start:.3f} seconds")

    

    parameter_names = {
        "module__hidden_dim": "hidden_dim",
        "module__dropout": "dropout",
        "lr": "learning_rate",
        "optimizer__weight_decay": "weight_decay",
        "max_epochs": "num_epochs",
    }
    defaults = {
        "hidden_dim": hidden_dim,
        "dropout": dropout,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "num_epochs": num_epochs,
    }
    if isinstance(param_grid, dict):
        param_grid = [param_grid]

    normalised_grid = []
    for candidate_grid in param_grid:
        normalised_candidate = {}
        for name, values in candidate_grid.items():
            name = parameter_names.get(name, name)
            if name not in defaults:
                raise ValueError(f"Unsupported hyperparameter: {name}")
            normalised_candidate[name] = values
        normalised_grid.append(normalised_candidate)

    parameter_candidates = list(ParameterGrid(normalised_grid or [{}]))
    if inner_cv < 2:
        raise ValueError("inner_cv must be at least 2")
    if inner_cv > len(training_set):
        raise ValueError("inner_cv cannot exceed the number of training examples")

    labels = np.asarray(training_labels)
    class_counts = np.bincount(labels)
    if len(class_counts) == 0 or class_counts[class_counts > 0].min() < inner_cv:
        raise ValueError("inner_cv cannot exceed the number of examples in the smallest class")

    splitter = StratifiedKFold(n_splits=inner_cv, shuffle=True, random_state=random_state)
    fold_splits = list(splitter.split(np.zeros(len(labels)), labels))
    best_score = float("inf")
    best_parameters = defaults.copy()
    print(f"Inner {inner_cv}-fold hyperparameter search with n_jobs={n_jobs}...")

    worker_arguments = [
        (   
            mmlp,
            dict(defaults, **candidate),
            training_set if retrain_encoder else None,
            None if retrain_encoder else X_train_sparse,
            labels,
            fold_splits,
            num_classes,
            None if retrain_encoder else len(encoder),
            None if retrain_encoder else encoder,
            EncoderClass if retrain_encoder else None,
            encoderdict if retrain_encoder else {},
        )
        for candidate in parameter_candidates
    ]

    if n_jobs == 0:
        raise ValueError("n_jobs cannot be 0")
    if n_jobs < 0:
        import os
        worker_count = os.cpu_count() or 1
    else:
        worker_count = n_jobs


    if worker_count == 1:
        results = [_evaluate_inner_candidate(*arguments) for arguments in worker_arguments]
    else:
        from joblib import Parallel, delayed

        results = Parallel(n_jobs=worker_count, backend="loky")(
            delayed(_evaluate_inner_candidate)(*arguments)
            for arguments in worker_arguments
        )
    training_text = [text for text, label in training_set]
    print(type(training_text))

    for parameters, mean_loss in results:
        print(f"Parameters {parameters} | inner val loss: {mean_loss:.4f}")
        if mean_loss < best_score:
            best_score = mean_loss
            best_parameters = parameters

    if retrain_encoder:
        training_text = [text for text, label in training_set]
        # print(training_text)
        # print(training_set)
        print(type(training_text))
        encoder = EncoderClass(
            tokenizer = train_tokenizer(
                training_text,
                **encoderdict,
            ),
            corpus = training_text
        )
        training_vectors = [encoder(text) for text, label in training_set]
        validation_vectors = [encoder(text) for text, label in validation_set]
        X_train_sparse = csr_matrix(training_vectors)
        X_val_sparse = csr_matrix(validation_vectors)

    train_dataset = SparseDataset(X_train_sparse, training_labels)
    val_dataset = SparseDataset(X_val_sparse, validation_labels)
    train_loader = DataLoader(
        train_dataset,
        batch_size=best_parameters["batch_size"],
        shuffle=True,
        collate_fn=sparse_collate,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=best_parameters["batch_size"],
        shuffle=False,
        collate_fn=sparse_collate,
    )
    model = SparseMMLP(
        input_dim=len(encoder),
        hidden_dim=best_parameters["hidden_dim"],
        num_classes=num_classes,
        dropout=best_parameters["dropout"],
    ).to(device) if mmlp else SparseMLP(
        input_dim=len(encoder),
        hidden_dim=best_parameters["hidden_dim"],
        num_classes=num_classes,
        dropout=best_parameters["dropout"],
    ).to(device)
    criterion = nn.NLLLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=best_parameters["learning_rate"],
        weight_decay=best_parameters["weight_decay"],
    )

    metrics = []
    # for epoch in range(best_parameters["num_epochs"]):
    #     train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device)
    #     val_metrics, predictions = evaluate(model, val_loader, criterion, device)
    #     metrics.append({"train_metrics": train_metrics, "val_metrics": val_metrics})
    #     print(
    #         f"Epoch {epoch + 1:02d} | "
    #         f"train loss: {train_metrics['loss']:.4f} | "
    #         f"train acc: {train_metrics['accuracy']:.4f} | "
    #         f"outer val loss: {val_metrics['loss']:.4f} | "
    #         f"outer val acc: {val_metrics['accuracy']:.4f}"
    #     )

    # best_epoch = best_parameters["num_epochs"] - 1
    # torch.save(model.state_dict(), f"{save_to}.pt")
    # with open(f"{save_to}.json", "w", encoding="utf-8") as f:
    #     json.dump({
    #         "input_dim": len(encoder),
    #         "hidden_dim": best_parameters["hidden_dim"],
    #         "num_classes": num_classes,
    #         "dropout": best_parameters["dropout"],
    #     }, f, indent=2, ensure_ascii=False)
    for epoch in tqdm(range(best_parameters["num_epochs"])):
        # Train the model for one epoch
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device)
        # Evaluate the model on the validation set
        val_metrics, preds = evaluate(model, val_loader, criterion, device)
        # Store the metrics for this epoch
        metrics.append({"train_metrics": train_metrics, "val_metrics": val_metrics})

        print(
            f"Epoch {epoch + 1:02d} | "
            f"train loss: {train_metrics['loss']:.4f} | "
            f"train acc: {train_metrics['accuracy']:.4f} | "
            f"val loss: {val_metrics['loss']:.4f} | "
            f"val acc: {val_metrics['accuracy']:.4f}"
            )
        
        # Saving the best model according to validation loss
        if epoch != 0:
            if val_metrics["loss"] <= best_val_loss:
                best_val_loss = val_metrics["loss"]
                torch.save(model.state_dict(), f"{save_to}.pt")
                best_epoch = epoch
                predictions = preds
        else:
            best_val_loss = val_metrics["loss"]
            torch.save(model.state_dict(), f"{save_to}.pt")
            best_epoch = epoch
            predictions = preds

    time_training = time.time()
    print(f"Best parameters: {best_parameters}")
    print(f"Training done! Time taken: {time_training - time_document_to_vector:.3f} seconds")

    return metrics, best_epoch, predictions


def load(path, device="cpu"):
    """
    Reconstruct the model architecture (using {path}.json) and load trained weights (using {path}.pt) and return it.
    """

    with open(f"{path}.json", "r", encoding="utf-8") as f:
        args = json.load(f)
    
    with open(f"{path}.pt", "rb") as f:
        state_dict = torch.load(f)
        
    if isinstance(device, str):
        device = torch.device(device)
        
    model = SparseMLP(**args)
    model.load_state_dict(state_dict)
    model = model.to(device)
    return model


def predict_probs(texts, model, encoder, batch_size=10):
    """
    Return a numpy array with shape [num_texts, num_classes] with the pmfs of the input texts as predicted by the model.
    
    texts: list of texts, each a string
    model: trained model
    encoder: encoder used in training
    batch_size: how many at a time
    """
    
    device = model.weight.device
        
    vectors = [encoder(x) for x in texts]
    labels = [0 for _ in texts]

    x_sparse = csr_matrix(vectors)

    dl = DataLoader(
        SparseDataset(x_sparse, labels),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=sparse_collate,
    )
    
    model.eval()  # disable dropout / training behaviour
    
    all_probs = []
    with torch.no_grad():  # no gradients needed for evaluation

        for x_batch, y_batch in dl:
            # move the batch to the device
            x_batch = x_batch.to(device) 
            y_batch = y_batch.to(device)
            logprobs = model(x_batch) # forward pass
            # accumulate metrics; also keep predictions for later use
            all_probs.extend(logprobs.exp().detach().cpu().numpy())
            
    return np.array(all_probs)