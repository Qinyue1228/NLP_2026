import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence 
from torch.nn.utils.rnn import pad_sequence 
import time
from tqdm.auto import tqdm
from collections import defaultdict, OrderedDict
from pathlib import Path
import itertools
import json

from . import clslib


class LabelledCorpus(Dataset):
    def __init__(self, X, y):
        """Receives a list of variable-length token-id sequences and labels."""
        self.X = [torch.tensor(x, dtype=torch.long) for x in X]
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def collate_labelled_texts(batched_pairs, padding_value=0):
    X_rows, y_rows = zip(*batched_pairs)    
    X_batch = pad_sequence(X_rows, batch_first=True, padding_value=padding_value)
    y_batch = torch.stack(y_rows)
    return X_batch, y_batch


def training_one_epoch(model, dataloader, optimizer, criterion, device="cpu"):
    if isinstance(device, str):
        device = torch.device(device)
        
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        logprobs = model(X_batch)
        loss = criterion(logprobs, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * y_batch.size(0)
        predictions = logprobs.argmax(dim=1)
        total_correct += (predictions == y_batch).sum().item()
        total_examples += y_batch.size(0)

    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
    }

def evaluate(model, dataloader, criterion, device="cpu"):
    if isinstance(device, str):
        device = torch.device(device)
        
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    preds_list = []
    labels_list = []
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logprobs = model(X_batch)
            loss = criterion(logprobs, y_batch)

            total_loss += loss.item() * y_batch.size(0)
            predictions = logprobs.argmax(dim=1)
            total_correct += (predictions == y_batch).sum().item()
            total_examples += y_batch.size(0)
            preds_list.append(predictions.tolist())
            labels_list.append(y_batch.tolist())

    return {
        "predictions": preds_list,
        "labels": labels_list,
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
    }

def train(
    training_set, 
    validation_set, 
    tokenizer: clslib.Tokenizer,    
    emb_model: clslib.PretrainedEmbeddings,
    num_classes: int,    
    enc_type: str,
    save_to="temp",    
    enc_kwargs={'freeze_embeddings': True},
    regressor_kwargs={'hidden_size1': 50, 'hidden_size2': 50, 'dropout': 0.},    
    opt_kwargs={'learning_rate': 1e-3, 'weight_decay': 1e-4, 'batch_size': 64, 'num_epochs': 20},    
    device='cpu'
):
    """Receives a training set and a validation set and returns a trained neural network model"""

    # 6
    
    time_start = time.time()

    print("Tokenizing texts...")
    # Convert the training and validation sets to vectors
    training_vectors = [tokenizer.ids(text) for text, label in training_set]
    training_labels = [int(label) for text, label in training_set]
    validation_vectors = [tokenizer.ids(text) for text, label in validation_set]
    validation_labels = [int(label) for text, label in validation_set]

    time_tokenizing = time.time()
    print(f"Tokenization done! Time taken: {time_tokenizing - time_start:.3f} seconds")

    pad_id = emb_model.vocab.index(emb_model.pad_token)    
    
    train_dataset = LabelledCorpus(training_vectors, training_labels)
    val_dataset = LabelledCorpus(validation_vectors, validation_labels)

    train_loader = DataLoader(
        train_dataset,
        batch_size=opt_kwargs['batch_size'],
        shuffle=True,
        collate_fn=lambda batch: collate_labelled_texts(batch, padding_value=pad_id),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=opt_kwargs['batch_size'],
        shuffle=False,
        collate_fn=lambda batch: collate_labelled_texts(batch, padding_value=pad_id),
    )

    time_dataloader = time.time()
    print(f"Dataloader done! Time taken: {time_dataloader - time_tokenizing:.3f} seconds")

    if isinstance(device, str):
        device = torch.device(device)

    # save architecture-defining arguments
    enc_kwargs['padding_idx'] = pad_id
    regressor_kwargs['num_classes'] = num_classes    
    args = {        
        'enc_kwargs': enc_kwargs,
        'regressor_kwargs': regressor_kwargs,
        'enc_type': enc_type,
    }
    with open(f"{save_to}.json", "w", encoding="utf-8") as f:        
        json.dump(args, f, indent=2, ensure_ascii=False) 

    if enc_type == 'embed-and-pool':
        encoder = clslib.EmbedAndPoolEncoder(
            embedding_matrix=torch.tensor(emb_model.vectors, dtype=torch.float32),
            **enc_kwargs
         )
    elif enc_type == 'lstm':
         encoder = clslib.LSTMEncoder(
             embedding_matrix=torch.tensor(emb_model.vectors, dtype=torch.float32), 
             **enc_kwargs
         )
    else:
        raise ValueError(f"I do not know this encoder type: {enc_type}")    

    model = clslib.SoftmaxRegressor(        
        encoder=encoder,
        **regressor_kwargs
    ).to(device)
    
    criterion = nn.NLLLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt_kwargs['learning_rate'], weight_decay=opt_kwargs['weight_decay'])

    metrics = []
    for epoch in tqdm(range(opt_kwargs['num_epochs'])):
        train_metrics = training_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
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
        else:
            best_val_loss = val_metrics["loss"]
            torch.save(model.state_dict(), f"{save_to}.pt")
            best_epoch = epoch
    
    time_training = time.time()
    print(f"Training done! Time taken: {time_training - time_dataloader:.3f} seconds")

    return metrics, best_epoch

    
def load(path, emb_model: clslib.PretrainedEmbeddings, device="cpu"):
    """
    Reconstruct the model architecture (using {path}.json) and load trained weights (using {path}.pt) and return it.
    """

    with open(f"{path}.json", "r", encoding="utf-8") as f:
        args = json.load(f)
    
    with open(f"{path}.pt", "rb") as f:
        state_dict = torch.load(f)
        
    if isinstance(device, str):
        device = torch.device(device)
        
    if args['enc_type'] == 'embed-and-pool':
        encoder = clslib.EmbedAndPoolEncoder(
            embedding_matrix=torch.tensor(emb_model.vectors, dtype=torch.float32),
            **args['enc_kwargs']
         )
    elif args['enc_type'] == 'lstm':
         encoder = clslib.LSTMEncoder(
             embedding_matrix=torch.tensor(emb_model.vectors, dtype=torch.float32), 
             **args['enc_kwargs']
         )
    else:
        raise ValueError(f"I do not know this encoder type: {args['enc_type']}")    

    model = clslib.SoftmaxRegressor(        
        encoder=encoder,
        **args['regressor_kwargs']
    )
    
    model.load_state_dict(state_dict)
    model = model.to(device)
    return model

def cross_validation(
    outcome_space,
    data_splits, 
    tokenizer: clslib.Tokenizer, 
    emb_model: clslib.PretrainedEmbeddings, 
    enc_type, 
    enc_kwargs,
    regressor_kwargs, 
    opt_kwargs, 
    device='cpu',
    save_to="tmp"):
    """
    Trains and evaluates k models using a k-fold data split for cross validation. 
    This function returns a dictionary with 
        - models: the paths to the k models trained (the model objects aren't returned, they are stored in disk)
        - predictions: the predictions by each of the k models
        - accuracies: the accuracy of each trained model on its corresponding validation partition
    
    data_splits: the training/validation folds
    encoder: an instance of Encoder
    model_option: for now this only supports 'logistic'
        which tells this function to use train_logistic_model (defined earlier in this notebook)
    save_to: folder where to store models
    smlp_args: dict with training arguments for smlp.train_neural_network_model
    """

    models, predictions, accuracies = OrderedDict(), OrderedDict(), OrderedDict()
    
    for fold in tqdm(data_splits.keys()):
        training_set = data_splits[fold]["training"]
        validation_set = data_splits[fold]["validation"]

        fold_folder = Path(f"{save_to}/fold-{fold}")
        fold_folder.mkdir(parents=True, exist_ok=True) 
        
        metrics, best_epoch = train(
            training_set, 
            validation_set, 
            tokenizer,
            emb_model,
            num_classes=len(outcome_space),            
            enc_type=enc_type,
            enc_kwargs=enc_kwargs,
            regressor_kwargs=regressor_kwargs,
            opt_kwargs=opt_kwargs,
            device=device,
            save_to=f'{fold_folder}/model'
        )
        models[fold] = f'{fold_folder}/model'
        accuracies[fold] = metrics[best_epoch]['val_metrics']['accuracy'] 
        predictions[fold] = np.array(list(itertools.chain.from_iterable(metrics[best_epoch]['val_metrics']['predictions'])))

    return {'models': models, 'predictions': predictions, 'accuracies': accuracies}    

def predict_logprobs(   
    model: clslib.SoftmaxRegressor,    
    tokenizer: clslib.Tokenizer,        
    texts,     
    batch_size: int,
    device='cpu'
):
    """Receives a training set and a validation set and returns a trained neural network model"""
        
    vectors = [tokenizer.ids(text) for text in texts]
    labels = [0 for _ in texts]

    pad_id = model.encoder.pad_idx

    dl = DataLoader(
        LabelledCorpus(vectors, labels),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_labelled_texts(batch, padding_value=pad_id),
    )

    if isinstance(device, str):
        device = torch.device(device)


    model.eval()

    preds_list = []
    labels_list = []
    logprobs_list = []
    with torch.no_grad():
        for x_batch, y_batch in dl:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            logprobs = model(x_batch)
            logprobs_list.extend(logprobs.cpu().numpy())
    
    return np.array(logprobs_list)