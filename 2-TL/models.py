
# General-Purpose Imports
import sys
import os
import warnings
import math
import json
import random
import numpy as np
from timeit import default_timer as timer
from matplotlib import pyplot as plt
from pathlib import Path

# PyTorch Libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.init import xavier_uniform_, zeros_, orthogonal_
from torch.utils.data import DataLoader, random_split, Dataset, TensorDataset

# Scikit-Learn Libraries
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, f1_score, roc_auc_score, r2_score

# Keras Libraries
if os.path.isdir("keras2cpp"):
    KERAS2CPP = True
else:
    KERAS2CPP = False
    print("WARNING: Keras2Cpp is either not installed or not present in the current directory.")
if KERAS2CPP:
    sys.path.append("keras2cpp/")
import tensorflow as tf
import gc
import tensorflow.keras as keras
import tensorflow.keras.backend as K
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Adagrad
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.activations import softmax, relu, tanh, sigmoid, elu, softplus, selu, softsign
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Dense, LSTM, GRU, SimpleRNN, Flatten, BatchNormalization, Dropout, Activation, Input, Layer
from tensorflow.keras.layers import Conv2D, Conv1D, MaxPooling2D, MaxPooling1D
from tensorflow.keras.layers import AveragePooling2D, AveragePooling1D 
from tensorflow.keras.layers import GlobalAveragePooling2D, GlobalAveragePooling1D, GlobalMaxPooling2D, GlobalMaxPooling1D
from tensorflow.keras.layers import SpatialDropout1D, SpatialDropout2D
from tensorflow.keras.initializers import GlorotUniform, Zeros, Orthogonal



# Custom Libraries
if KERAS2CPP:
    from keras2cpp import export_model



# Set random seeds
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
tf.random.set_seed(SEED)
GLOROTUNIFORM = GlorotUniform(seed=SEED)
ORTHOGONAL = Orthogonal(seed=SEED)


########################################################################################################################
# Global variables, functions, and classes
########################################################################################################################


optdict_keras = {'adam':Adam, 'sgd':SGD, 'rmsprop':RMSprop, 'adagrad':Adagrad}
actdict_keras = {
    'relu':relu, 'leakyrelu':LeakyReLU(alpha=0.1), 
    'sigmoid':sigmoid, 'tanh':tanh, 'softmax':softmax,
    'softplus':softplus, 'softsign':softsign,
    'elu':elu, 'selu':selu}
rnndict_keras = {'lstm':LSTM, 'gru':GRU, 'rnn':SimpleRNN}


actdict_pytorch = {
    'relu':nn.ReLU(), 'leakyrelu':nn.LeakyReLU(0.1), 'sigmoid':nn.Sigmoid(), 'tanh':nn.Tanh(),
    'softmax':nn.Softmax(dim=1), 'logsoftmax':nn.LogSoftmax(dim=1),
    'softplus':nn.Softplus(), 'softshrink':nn.Softshrink(),
    'elu':nn.ELU(), 'selu':nn.SELU(), 'softsign':nn.Softsign(), 'softmin':nn.Softmin(dim=1),
    'softmax2d':nn.Softmax2d()}

lossdict_pytorch = {
    "mse":nn.MSELoss, "crossentropy":nn.CrossEntropyLoss, "binary_crossentropy":nn.BCELoss,
    "categorical_crossentropy":nn.CrossEntropyLoss, "nll":nn.NLLLoss, "poisson":nn.PoissonNLLLoss,
    "kld":nn.KLDivLoss, "hinge":nn.HingeEmbeddingLoss, "l1":nn.L1Loss,
    "mae": nn.L1Loss, "l2":nn.MSELoss, "smoothl1":nn.SmoothL1Loss, "bce_with_logits":nn.BCEWithLogitsLoss
}
optdict_pytorch = {'adam':optim.Adam, 'sgd':optim.SGD, 'rmsprop':optim.RMSprop}

# When using CrossEntropyLoss, the output should include class incdices rather than one-hot encoded propbabilities,
# unless two classes can be chosen at once for an input data.
# When using CrossEntropyLoss, during inference, logsoftmax should manually be applied.
# PoissonNLLLoss does not require logsoftmax before it.
# CrossEntropyLoss without output activation is equal to NLLLoss with LogSoftmax output activation.
# BCELoss (binary crossentropy) requires sigmoid output activation. Values should be probabilities.
# Like NLLLoss, KLDivLoss also requires LogSoftmax output activations (it should be in the log space).
# That is, the input (y_pred) should be log probabilities (output of logsoftmax).
# If the target output is e.g. only sigmoid (not log space) then log_target=False should be set.
# If the target output is also the output of a logsoftmax (includes log probabilities) then log_target=True
# BceWithLogitsLoss is equal to Sigmoid and BCELoss. For Eval sigmoid is not needed, but for inference a sigmoid should
# manually be added, just like CrossEntropyLoss. 


def make_path(path:str):
    Path.mkdir(Path(path).parent, parents=True, exist_ok=True)
    return path



def plot_keras_model_history(history:dict, metrics:list=['loss','val_loss'], 
                             fig_title:str='model loss', saveto:str=None, figsize=(10,5)):
    plt.figure(figsize=figsize)
    plt.grid(True)
    plt.plot(history[metrics[0]])
    plt.plot(history[metrics[1]])
    plt.title(fig_title)
    plt.ylabel(metrics[0])
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    if saveto:
        plt.savefig(make_path(saveto), dpi=600)
    
        


def compile_keras_model(model, _batchsize:int, _learnrate:float, _optimizer:str, _loss:str, _metrics:list, 
                          _optimizerparams:dict=None, _learnrate_decay_gamma:float=None, num_samples:int=None):
    if _learnrate_decay_gamma:
        itersPerEpoch = (num_samples//_batchsize) if num_samples else 1
        sch = ExponentialDecay(initial_learning_rate=_learnrate, 
        decay_steps=itersPerEpoch, decay_rate=_learnrate_decay_gamma)
        lr = sch
    else:
        lr = _learnrate
    if _optimizerparams:
        optparam = _optimizerparams
        opt = optdict_keras[_optimizer](learning_rate=lr, **optparam)
    else:
        opt = optdict_keras[_optimizer](learning_rate=lr)
    model.compile(optimizer=opt, loss=_loss, metrics=_metrics)
    


def fit_keras_model(model, x_train, y_train, x_val, y_val, 
    _batchsize:int, _epochs:int, _callbacks:list, verbose:bool=True, **kwargs):
    while True:
        try:
            history = model.fit(x_train, y_train, batch_size=_batchsize, epochs=_epochs, 
                validation_data=(x_val, y_val), verbose=verbose, 
                callbacks=_callbacks, **kwargs)
            break
        except Exception as e:
            print(e)
            print(("\nTraining failed with batchsize={}. "+\
                "Trying again with a lower batchsize...").format(_batchsize))
            _batchsize = _batchsize // 2
            if _batchsize < 2:
                raise ValueError("Batchsize too small. Training failed.")
    return history



def save_keras_model(model, history:dict, path:str, hparams:dict):
    try:
        model.save(make_path(path))
        for key in history:
            hparams[key] = history[key]
        jstr = json.dumps(hparams, indent=4)
        with open(path+"/hparams.json", "w") as f:
            f.write(jstr)
    except Exception as e:
        print(e)
        print("Cannot serialize Keras model.")


        
def export_keras_model(model, path:str):
    if KERAS2CPP:
        try:
            export_model(model, make_path(path))
            print("Model exported successfully.")
        except Exception as e1:
            print(e1)
            print("Cannot export Keras model using keras2cpp on the fly. Will try sequentializing the model layers...")
            try:
                net = Sequential(model.layers)
                export_model(net, make_path(path))
                print("Model exported successfully.")
            except Exception as e2:
                print(e2)
                print("Cannot export model using Keras2Cpp.")
    else:
        print("Cannot export model using keras2cpp. Keras2cpp is not installed.")



        
def test_keras_model(model_class):
    print("Constructing model...\n")
    model = model_class()
    print("Summary of model:")
    model.summary()
    print("\nGenerating random dataset...\n")
    (x_train, y_train) = generate_sample_batch(model)
    (x_val, y_val) = generate_sample_batch(model)
    print("\nTraining model...\n")
    model.train(x_train, x_val, y_train, y_val, 
                verbose=True, saveto="test_"+model.hparams["model_name"], 
                export="test_"+model.hparams["model_name"]+".model")
    
    print("\nEvaluating model...\n")
    model.evaluate(x_val, y_val, verbose=True)
    print("Done.")
    



def generate_sample_batch(model):
    x = np.random.rand(*model.batch_input_shape).astype(np.float32)
    y = np.random.rand(*model.batch_output_shape).astype(np.float32)
    return (x,y)


def test_pytorch_model(model_class, saveto_homedir='./', **kwargs):
    print("Constructing model...\n")
    model = model_class(**kwargs)
    print("Summary of model:")
    print(model)
    print("\nGenerating random dataset...\n")
    (x_train, y_train) = generate_sample_batch(model)
    (x_val, y_val) = generate_sample_batch(model)
    x_train_t = torch.Tensor(x_train)
    y_train_t = torch.Tensor(y_train)
    x_val_t = torch.Tensor(x_val)
    y_val_t = torch.Tensor(y_val)
    trainset = TensorDataset(x_train_t, y_train_t)
    validset = TensorDataset(x_val_t, y_val_t)
    dataset = (trainset, validset)
    print("\nTraining model...\n")
    model.train_model(dataset, verbose=True, script_before_save=True, 
                      saveto=os.path.join(saveto_homedir,"dummy_%s.pt"%model.hparams["model_name"]))
    print("\nEvaluating model...\n")
    # Nothing is done here yet.
    print("Done.")
    



def train_pytorch_model(model, dataset, batch_size:int, loss_str:str, optimizer_str:str, 
    optimizer_params:dict=None, loss_function_params:dict=None, learnrate:float=0.001, 
    learnrate_decay_gamma:float=None, epochs:int=10, validation_patience:int=10000, validation_data:float=0.1, 
    verbose:bool=True, script_before_save:bool=True, saveto:str=None, num_workers=0):
    """Train a Pytorch model, given some hyperparameters.

    ### Args:
        - `model` (`torch.nn`): A torch.nn model
        - `dataset` (`torch.utils.data.Dataset`): Dataset object to be used
        - `batch_size` (int): Batch size
        - `loss_str` (str): Loss function to be used. Options are "mse", "mae", "crossentropy", etc.
        - `optimizer_str` (str): Optimizer to be used. Options are "sgd", "adam", "rmsprop", etc.
        - `optimizer_params` (dict, optional): Parameters for the optimizer.
        - `loss_function_params` (dict, optional): Parameters for the loss function.
        - `learnrate` (float, optional): Learning rate. Defaults to 0.001.
        - `learnrate_decay_gamma` (float, optional): Learning rate exponential decay rate. Defaults to None.
        - `epochs` (int, optional): Number of epochs. Defaults to 10.
        - `validation_patience` (int, optional): Number of epochs to wait before stopping training. Defaults to 10000.
        - `validation_data` (float, optional): Fraction of the dataset to be used for validation. Defaults to 0.1.
        - `verbose` (bool, optional): Whether to print progress. Defaults to True.
        - `script_before_save` (bool, optional): Use TorchScript for serializing the model. Defaults to True.
        - `saveto` (str, optional): Save PyTorch model in path. Defaults to None.
        - `num_workers` (int, optional): Number of workers for the dataloader. Defaults to 0.
        

    ### Returns:
        - `model`: Trained PyTorch-compatible model
        - `history`: PyTorch model history dictionary, containing the following keys:
            - `training_loss`: List containing training loss values of epochs.
            - `validation_loss`: List containing validation loss values of epochs.
            - `learning_rate`: List containing learning rate values of epochs.
    """

    hist_training_loss = []
    hist_validation_loss = []
    hist_learning_rate = []
    hist_trn_metric = []
    hist_val_metric = []

    if torch.cuda.is_available(): torch.cuda.empty_cache()

    if "list" in type(dataset).__name__ or "tuple" in type(dataset).__name__:
        assert len(dataset)==2, "If dataset is a tuple, it must have only two elements, "+\
            "the training dataset and the validation dataset."
        trainset, valset = dataset
        num_val_data = int(len(valset))
        num_train_data = int(len(trainset))
        num_all_data = num_train_data + num_val_data
    else:
        num_all_data = len(dataset)
        num_val_data = int(validation_data*num_all_data)
        num_train_data = num_all_data - num_val_data
        (trainset, valset) = random_split(dataset, (num_train_data, num_val_data), 
            generator=torch.Generator().manual_seed(SEED))

    
    if verbose:
        print("Total number of data points:      %d"%num_all_data)
        print("Number of training data points:   %d"%num_train_data)
        print("Number of validation data points: %d"%num_val_data)
        
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    validloader = DataLoader(valset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    
    
    if verbose:
        print("Number of training batches:    %d"%len(trainloader))
        print("Number of validation batches:  %d"%len(validloader))
        print("Batch size:                    %d"%batch_size)
        for x,y in trainloader:
            print("Shape of x_train:", x.shape)
            print("Shape of y_train:", y.shape)
            break
        for x,y in validloader:
            print("Shape of x_val:", x.shape)
            print("Shape of y_val:", y.shape)
            break

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("selected device: ", device)
    
    model.to(device)
    
    loss_func = lossdict_pytorch[loss_str]
    if loss_function_params:
        criterion = loss_func(**loss_function_params)
    else:
        criterion = loss_func()
        
    
    optimizer_func = optdict_pytorch[optimizer_str]
    if optimizer_params:
        optimizer = optimizer_func(model.parameters(), lr=learnrate, **optimizer_params)
    else:
        optimizer = optimizer_func(model.parameters(), lr=learnrate)

    
    if learnrate_decay_gamma:
        if verbose:
            print("The learning rate has an exponential decay rate of %.5f."%learnrate_decay_gamma)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=learnrate_decay_gamma)
        lr_sch = True
    else:
        lr_sch = False
    
    # Find out if we're going to display any metric along with the loss or not.
    display_metrics = True
    classification = False
    regression = False
    if loss_str in ["binary_crossentropy", "bce_with_logits", "nll", "crossentropy", 
                    "categorical_crossentropy"]:
        classification = True
        regression = False
        trn_metric_name = "Acc"
        val_metric_name = "Val Acc"
    elif loss_str in ["mse", "l1", "l2", "mae"]:
        classification = False
        regression = True
        trn_metric_name = "R2"
        val_metric_name = "Val R2"
    else:
        classification = False
        regression = False
        display_metrics = False
    if verbose:
        if classification:
            print("Classification problem detected. We will look at accuracies.")
        elif regression:
            print("Regression problem detected. We will look at R2 scores.")
        else:
            print("We have neither classification nor regression problem. No metric will be displayed.")
    
                    
    # Preparing training loop
    num_training_batches = len(trainloader)
    num_validation_batches = len(validloader)
    
    progress_bar_size = 40
    ch = "█"
    intvl = num_training_batches/progress_bar_size;
    valtol = validation_patience if validation_patience else 10000
    minvalerr = 1000000.0
    badvalcount = 0
    
    # Commencing training loop
    tStart = timer()
    for epoch in range(epochs):
        
        tEpochStart = timer()
        epoch_loss_training = 0.0
        epoch_loss_validation = 0.0
        newnum = 0
        oldnum = 0
        trn_metric = 0.0
        val_metric = 0.0
    
        if verbose and epoch > 0: print("Epoch %3d/%3d ["%(epoch+1, epochs), end="")
        if verbose and epoch ==0: print("First epoch ...")
        
        ##########################################################################
        # Training
        if verbose and epoch==0: print("\nTraining phase ...")
        model.train()
        for i, data in enumerate(trainloader):
            # Fetching data
            seqs, targets = data[0].to(device), data[1].to(device)
            
            # Loss calculation
            predictions = model(seqs)
            loss = criterion(predictions, targets)
            epoch_loss_training += loss.item()
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Metrics calculation
            if display_metrics:
                with torch.no_grad():
                    if loss_str == "binary_crossentropy":
                        # Output layer already includes sigmoid.
                        class_predictions = (predictions > 0.5).float()
                    elif loss_str == "bce_with_logits":
                        # Output layer does not include sigmoid. Sigmoid is a part of the loss function.
                        class_predictions = (torch.sigmoid(predictions) > 0.5).float()
                    elif loss_str == "nll":
                        # Output layer already includes log_softmax.
                        class_predictions = torch.argmax(predictions, dim=1, keepdim=True).float()
                    elif loss_str in ["crossentropy", "categorical_crossentropy"]:
                        # Output layer does not have log_softmax. It is implemented as a part of the loss function.
                        class_predictions = torch.argmax(torch.log_softmax(predictions, dim=1), dim=1)#, keepdim=True)   

                    if classification:
                        if verbose and i==0 and epoch ==0: 
                            print("Shape of class_predictions: ", class_predictions.shape)
                            print("Shape of targets:           ", targets.shape)
                        
                        # Calculate accuracy
                        correct = (class_predictions == targets).float().sum().item()
                        trn_metric += correct
                        if verbose and epoch==0: 
                            print("Number of correct answers (this batch/total): %5d/%5d"%(correct, trn_metric))
                        
                        # Calculate F1 score
                        # f1 = f1_score(targets.cpu().numpy(), class_predictions.cpu().numpy(), average="macro")
                        # Calculate ROC AUC
                        # auc = roc_auc_score(targets.cpu().numpy(), class_predictions.cpu().numpy(), average="macro")
                    elif regression:
                        if verbose and i==0 and epoch==0: 
                            print("Shape of predictions: ", predictions.shape)
                            print("Shape of targets:     ", targets.shape)
                        # Calculate r2_score
                        trn_metric += r2_score(targets.cpu().numpy(), predictions.cpu().numpy())
                    
                    
            # Visualization of progressbar
            if verbose and epoch > 0:
                newnum = int(i/intvl)
                if newnum > oldnum:
                    print((newnum-oldnum)*ch, end="")
                    oldnum = newnum 
        
        if lr_sch:
            scheduler.step()
        
        epoch_loss_training /= num_training_batches
        if verbose and epoch==0: print("Epoch loss (training): %.5f"%epoch_loss_training)
        hist_training_loss.append(epoch_loss_training)
        
        if display_metrics:
            if classification:
                trn_metric /= num_train_data
            else:
                trn_metric /= num_training_batches
            if verbose and epoch==0: print("Epoch metric (training): %.5f"%trn_metric)
            hist_trn_metric.append(trn_metric)
        
        
        if verbose and epoch > 0: print("] ", end="")
        
           
        ##########################################################################
        # Validation
        if verbose and epoch==0: print("\nValidation phase ...")
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(validloader):
                seqs, targets = data[0].to(device), data[1].to(device)
                predictions = model(seqs)
                loss = criterion(predictions, targets)
                epoch_loss_validation += loss.item()
                
                # Do prediction for metrics
                if display_metrics:
                    if loss_str == "binary_crossentropy":
                        # Output layer already includes sigmoid.
                        class_predictions = (predictions > 0.5).float()
                    elif loss_str == "bce_with_logits":
                        # Output layer does not include sigmoid. Sigmoid is a part of the loss function.
                        class_predictions = (torch.sigmoid(predictions) > 0.5).float()
                    elif loss_str == "nll":
                        # Output layer already includes log_softmax.
                        class_predictions = torch.argmax(predictions, dim=1, keepdim=True).float()
                    elif loss_str in ["crossentropy", "categorical_crossentropy"]:
                        # Output layer does not have log_softmax. It is implemented as a part of the loss function.
                        class_predictions = \
                            torch.argmax(torch.log_softmax(predictions, dim=1), dim=1)#, keepdim=True).float()    
                
                    if classification:
                        if verbose and i==0 and epoch ==0: 
                            print("Shape of class_predictions: ", class_predictions.shape)
                            print("Shape of targets:           ", targets.shape)
                        # Calculate accuracy
                        correct = (class_predictions == targets).float().sum().item()
                        val_metric += correct
                        if verbose and epoch==0: 
                            print("Number of correct answers (this batch/total): %5d/%5d"%(correct, val_metric))
                        # Calculate F1 score
                        # f1 = f1_score(targets.cpu().numpy(), class_predictions.cpu().numpy(), average="macro")
                        # Calculate ROC AUC
                        # auc = roc_auc_score(targets.cpu().numpy(), class_predictions.cpu().numpy(), average="macro")
                    elif regression:
                        if verbose and i==0 and epoch==0: 
                            print("Shape of predictions: ", predictions.shape)
                            print("Shape of targets:     ", targets.shape)
                        # Calculate r2_score
                        val_metric += r2_score(targets.cpu().numpy(), predictions.cpu().numpy())
                    
        epoch_loss_validation /= num_validation_batches
        if verbose and epoch==0: print("Epoch loss (validation): %.5f"%epoch_loss_validation)
        hist_validation_loss.append(epoch_loss_validation)
                
        if display_metrics:
            if classification:
                val_metric /= num_val_data
            else:
                val_metric /= num_validation_batches
            if verbose and epoch==0: print("Epoch metric (validation): %.5f"%val_metric)
            hist_val_metric.append(val_metric)
        
        if lr_sch:
            hist_learning_rate.append(scheduler.get_last_lr())
        else:
            hist_learning_rate.append(learnrate)
        
        
        ##########################################################################
        # Post Processing Training Loop            
        tEpochEnd = timer()
        if verbose:
            if display_metrics:
                print("Loss: %5.4f |Val Loss: %5.4f |%s: %5.4f |%s: %5.4f | %6.3f s" % (
                    epoch_loss_training, epoch_loss_validation, trn_metric_name, trn_metric,
                    val_metric_name, val_metric, tEpochEnd-tEpochStart))
            else:
                print("Loss: %5.4f |Val Loss: %5.4f | %6.3f s" % (
                    epoch_loss_training, 
                    epoch_loss_validation, tEpochEnd-tEpochStart))
        

        # Checking for early stopping
        if epoch_loss_validation < minvalerr:
            minvalerr = epoch_loss_validation
            badvalcount = 0
        else:
            badvalcount += 1
            if badvalcount > valtol:
                if verbose:
                    print("Validation loss not improved for more than %d epochs."%badvalcount)
                    print("Early stopping criterion with validation loss has been reached. " + 
                        "Stopping training at %d epochs..."%epoch)
                break
    # End for loop
    
    
    ##########################################################################
    # Epilogue
    tFinish = timer()
    if verbose:        
        print('Finished Training.')
        print("Training process took %.2f seconds."%(tFinish-tStart))
    if saveto:
        try:
            if verbose: print("Saving model...")
            if script_before_save:
                example,_ = next(iter(trainloader))
                example = example[0,:].unsqueeze(0)
                model.cpu()
                with torch.no_grad():
                    traced = torch.jit.trace(model, example)
                    traced.save(saveto)
            else:
                with torch.no_grad():
                    torch.save(model, saveto)
        except Exception as e:
            if verbose:
                print(e)
                print("Failed to save the model.")
        if verbose: print("Done.")
        
    torch.cuda.empty_cache()
    
    history = {
        'training_loss':hist_training_loss, 
        'validation_loss':hist_validation_loss, 
        'learning_rate':hist_learning_rate}
    if display_metrics:
        history["training_metrics"] = hist_trn_metric
        history["validation_metrics"] = hist_val_metric
    
    return history






def evaluate_pytorch_model(model, dataset, loss_str:str, loss_function_params:dict=None,
    batch_size:int=16, device_str:str="cuda", verbose:bool=True, num_workers:int=0):
    """
    Evaluates a PyTorch model on a dataset.
    
    ### Parameters
    
    `model` (`torch.nn.Module`): The model to evaluate.
    `dataset` (`torch.utils.data.Dataset`): The dataset to evaluate the model on.
    `loss_str` (str): The loss function to use when evaluating the model.
    `loss_function_params` (dict, optional) : Parameters to pass to the loss function.
    `batch_size` (int, optional) : The batch size to use when evaluating the model. Defaults to 16.
    `device_str` (str, optional) : The device to use when evaluating the model. Defaults to "cuda".
    `verbose` (bool, optional) : Whether to print out the evaluation metrics. Defaults to True.
    `num_workers` (int, optional) : The number of workers to use when evaluating the model. Defaults to 0.
    
    
    ### Returns
    
    A dictionary containing the evaluation metrics, including "loss" and "metrics" in case any metric is available.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    if verbose: print("Preparing data...")
    testloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    num_batches = len(testloader)
    
    if "cuda" in device_str:
        device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    
    print("selected device: ", device)
    model.to(device)
    
    loss_func = lossdict_pytorch[loss_str]
    if loss_function_params:
        criterion = loss_func(**loss_function_params)
    else:
        criterion = loss_func()
    
    
    display_metrics = True
    classification = False
    regression = False
    if loss_str in ["binary_crossentropy", "bce_with_logits", "nll", "crossentropy", "categorical_crossentropy"]:
        classification = True
        regression = False
        metric_name = "Accuracy"
    elif loss_str in ["mse", "l1", "l2", "mae"]:
        classification = False
        regression = True
        metric_name = "R2-Score"
    else:
        classification = False
        regression = False
        display_metrics = False
        
    progress_bar_size = 20
    ch = "█"
    intvl = num_batches/progress_bar_size;
    if verbose: print("Evaluating model...")
    model.eval()
    newnum = 0
    oldnum = 0
    totloss = 0.0
    if verbose: print("[", end="")
    if display_metrics:
        val_metric = 0.0
    with torch.no_grad():
        for i, data in enumerate(testloader):
            inputs, targets = data[0].to(device), data[1].to(device)
            predictions = model(inputs)
            loss = criterion(predictions, targets)
            totloss += loss.item()
            
            # Do prediction for metrics
            if display_metrics:
                if loss_str == "binary_crossentropy":
                    # Output layer already includes sigmoid.
                    class_predictions = (predictions > 0.5).float()
                elif loss_str == "bce_with_logits":
                    # Output layer does not include sigmoid. Sigmoid is a part of the loss function.
                    class_predictions = (torch.sigmoid(predictions) > 0.5).float()
                elif loss_str == "nll":
                    # Output layer already includes log_softmax.
                    class_predictions = torch.argmax(predictions, dim=1, keepdim=True).float()
                elif loss_str in ["crossentropy", "categorical_crossentropy"]:
                    # Output layer does not have log_softmax. It is implemented as a part of the loss function.
                    class_predictions = \
                        torch.argmax(torch.log_softmax(predictions, dim=1), dim=1, keepdim=True).float()    
            
                if classification:
                    # Calculate accuracy
                    val_metric += (class_predictions == targets).float().mean()
                    # Calculate F1 score
                    # f1 = f1_score(targets.cpu().numpy(), class_predictions.cpu().numpy(), average="macro")
                    # Calculate ROC AUC
                    # auc = roc_auc_score(targets.cpu().numpy(), class_predictions.cpu().numpy(), average="macro")
                elif regression:
                    # Calculate r2_score
                    val_metric += r2_score(targets.cpu().numpy(), predictions.cpu().numpy())
                    
            # Visualization of progressbar
            if verbose:
                newnum = int(i/intvl)
                if newnum > oldnum:
                    print((newnum-oldnum)*ch, end="")
                    oldnum = newnum 
        
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
                        
    if verbose: print("] ", end="")
                    
    if display_metrics:
        val_metric /= num_batches
    
                
    totloss /= num_batches
    if verbose:
        if display_metrics:
            print("Loss: %5.4f | %s: %5.4f" % (totloss, metric_name, val_metric))
        else:
            print("Loss: %5.4f" % totloss)
            
    if verbose: print("Done.")
    
    d = {"loss":totloss}
    if display_metrics:
        d["metrics"] = val_metric
    
    return d
            
            

################################
def predict_pytorch_model(model, dataset, loss_str:str, batch_size:int=16, device_str:str="cuda", 
    return_in_batches:bool=True, return_inputs:bool=False, return_raw_predictions:bool=False, 
    verbose:bool=True, num_workers:int=0):
    """
    Predicts the output of a pytorch model on a given dataset.

    ### Args:
        - `model` (`torch.nn.Module`): The PyTorch model to use.
        - `dataset` (`torch.utils.data.Dataset`): Dataset containing the input data
        - `loss_str` (str): Loss function used when training. 
            Used only for determining whether a classification or a regression model is used.
        - `batch_size` (int, optional): Batch size to use when evaluating the model. Defaults to 16.
        - `device_str` (str, optional): Device to use when performing inference. Defaults to "cuda".
        - `return_in_batches` (bool, optional): Whether the predictions should be batch-separated. Defaults to True.
        - `return_inputs` (bool, optional): Whether the output should include the inputs as well. Defaults to False.
        - `return_raw_predictions` (bool, optional): Whether raw predictions should also be returned. Defaults to False.
        - `verbose` (bool, optional): Verbosity of the function. Defaults to True.
        - `num_workers` (int, optional): Number of workers to use when loading the data. Defaults to 0.

    ### Returns:
        List: A List containing the output predictions, and optionally, the inputs and raw predictions.
        
    ### Notes:
        - If `return_in_batches` is True, the output will be a list of lists. output[i] contains the i'th batch.
        - If `return_inputs` is true, the first element of the output information will be the inputs.
        - If `return_raw_predictions` is true, the second element of the output information will be the raw predictions.
            Please note that this is only meaningful for classification problems. Otherwise, predictions will only
            include raw predictions. For classification problems, if this setting is True, the third element of the
            output information will be the class predictions.
        - "output information" here is a list containing [input, raw_predictions, class_predictions].
            For non-classification problems, "output information" will only contain [input, raw_predictions].
            If `return_inputs` is False, the first element of the output information will be omitted; [raw_predictions].
            If `return_in_batches` is True, the output will be a list of "output information" for every batch.
            Otherwise, the output will be one "output information" for the whole dataset.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    if verbose: print("Preparing data...")
    testloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    num_batches = len(testloader)
    
    if "cuda" in device_str:
        device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    
    print("selected device: ", device)
    model.to(device)
    
    
    if loss_str in ["binary_crossentropy", "bce_with_logits", "nll", "crossentropy", "categorical_crossentropy"]:
        classification = True
    else:
        classification = False
    
    
    
    output_list = []
    
        
    progress_bar_size = 20
    ch = "█"
    intvl = num_batches/progress_bar_size;
    if verbose: print("Performing Prediction...")
    model.eval()
    newnum = 0
    oldnum = 0
    if verbose: print("[", end="")
    with torch.no_grad():
        for i, data in enumerate(testloader):
            inputs = data[0].to(device)
            predictions = model(inputs)
            
            # Do prediction
            if classification:
                if loss_str == "binary_crossentropy":
                    # Output layer already includes sigmoid.
                    class_predictions = (predictions > 0.5).float()
                elif loss_str == "bce_with_logits":
                    # Output layer does not include sigmoid. Sigmoid is a part of the loss function.
                    class_predictions = (torch.sigmoid(predictions) > 0.5).float()
                elif loss_str == "nll":
                    # Output layer already includes log_softmax.
                    class_predictions = torch.argmax(predictions, dim=1, keepdim=True).float()
                elif loss_str in ["crossentropy", "categorical_crossentropy"]:
                    # Output layer does not have log_softmax. It is implemented as a part of the loss function.
                    class_predictions = \
                        torch.argmax(torch.log_softmax(predictions, dim=1), dim=1, keepdim=True).float()    
            
            
            # Add batch predictions to output dataset
            obatch = []
            if return_inputs:
                obatch.append(inputs.cpu().numpy())
            if classification:
                if return_raw_predictions:
                    obatch.append(predictions.cpu().numpy())
                obatch.append(class_predictions.cpu().numpy())
            else:
                obatch.append(predictions.cpu().numpy())
                
            if return_in_batches:
                output_list.append(obatch)
            elif i==0:
                output_array = obatch
            else:
                for j in range(len(obatch)):
                    output_array[j] = np.append(output_array[j], obatch[j], axis=0)
            
              
            # Visualization of progressbar
            if verbose:
                newnum = int(i/intvl)
                if newnum > oldnum:
                    print((newnum-oldnum)*ch, end="")
                    oldnum = newnum 
        
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
                        
    if verbose: print("] ")
    
    if return_in_batches:
        return output_list
    else:
        return output_array
    
    


########################################################################################################################
# Defined Classes
########################################################################################################################



# Perform garbage collection
class MyCustomCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()
        
        
# Deploy early stopping when performance reaches good values
class EarlyStopAtCriteria(keras.callbacks.Callback):
    def __init__(self, monitor='val_loss', mode='min', value=0.001):
        super(keras.callbacks.Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.mode = mode
    def on_epoch_end(self, epoch, logs=None):
        if self.mode == 'min':
            if logs.get(self.monitor) <= self.value:
                print("Early stopping performance criteria has been reached. Stopping training.")
                self.model.stop_training = True
        else:
            if logs.get(self.monitor) >= self.value:
                print("Early stopping performance criteria has been reached. Stopping training.")
                self.model.stop_training = True




class Keras_ANN:
    
    
    sample_hparams = {
        "model_name": "Keras_ANN",
        "input_size": 10,
        "output_size": 3,
        "width": "auto",
        "depth": 2,
        "hidden_activation": "relu",
        "output_activation": "softmax",
        "batchnorm": "before",
        "batchnorm_params": None,
        "dropout": 0.2,
        "learning_rate": 0.001,
        "learning_rate_decay_gamma": 0.99,
        "optimizer": "adam",
        "optimizer_params": {"epsilon": 1e-08},
        "batch_size": 32,
        "epochs": 2,
        "validation_tolerance_epochs": 2,
        "early_stopping_monitor": "val_loss",
        "early_stopping_mode": "min",
        "early_stopping_value": 0.001,
        "l2_reg": 0.0001,
        "loss_function": "categorical_crossentropy",
        "loss_function_params": None,
        "metrics": ["accuracy"],
        "checkpoint_path": "dummy_Keras_ANN_best_weights.h5"
    }
    
    
    def __init__(self, hparams:dict=None):
        """Typical Artificial Neural Network class, also known as multilayer perceptron.
        This class will create a fully connected feedforward artificial neural network.
        It can be used for classification, regression, etc.
        It basically encompasses enough options to build all kinds of ANNs with any number of 
        inputs, outputs, layers with custom or arbitrary width or depth, etc.
        Supports multiple activation functions for hidden layers and the output layer,
        but the activation function of the hidden layers are all the same.
        
        ### Usage

        `net = Keras_ANN(hparams)` where `hparams` is the dictionary of hyperparameters. It can include:

            - `input_size` (int): Number of inputs to the ANN, i.e. size of the input layer.
            - `output_size` (int): Number of outputs to predict, i.e. size of the output layer.

            - `width` ("auto"|int|list|array): Hidden layer width. "auto" decides automatically, 
                a number sets them all the same, and a list/array sets each hidden layer differently.
                If "auto", hidden layer widths will be set in such a way that the first half of the network will be the 
                encoder and the other half will be the decoder.
                Therefore, the first hidden layer will be twice as large as the input layer, 
                and every layer of the encoder will be twice as large as the previous one.
                In the decoder half, layer width will be halved until the output layer.
                Layer widths will be powers of two.
            - `depth` (int): Specifies the depth of the network (number of hidden layers).
                It must be specified unless `width` is provided as a list. Then the depth will be inferred form it.

            - `hidden_activation` (str): Activation of the hidden layers.
                Supported activations are lowercase names of the activations in Keras. LeakyReLU will have alpha=0.1.
            - `output_activation` (str): Activation of the output layer, if any. 
                Supported options are the same as before.
                **Note**: For classification problems, you might want to choose "sigmoid" or "softmax".
                **Note**: For regression problems, no activation is needed. It is by default linear, 
                unless you want to manually specify an activation.

            - `batchnorm` (str): If given, specifies where the batch normalization layer should be included: 
                `"before"` the activation, or `"after"` it.
                For activation functions such as **ReLU** and **sigmoid**, `"before"` is usually a better option. 
                For **tanh** and **LeakyReLU**, `"after"` is usually a better option.
            - `batchnorm_params` (dict): Parameters for the batch normalization layer.

            - `dropout` (float): If given, specifies the dropout rate after every 
                hidden layer. It should be a probability value between 0 and 1.

            - `learning_rate` (float): Initial learning rate of training.
            - `learning_rate_decay_gamma` (float): Exponential decay rate gamma for learning rate, if any.
            - `optimizer` (str): Optimizer, options are "sgd", "adam", "rmsprop" for now.
            - `optimizer_params` (dict): Additional parameters of the optimizer, if any.
            - `batch_size` (int): Minibatch size for training.
            - `epochs` (int): Maximum number of epochs for training.
            - `validation_tolerance_epochs` (int): Number of epochs to tolerate validation loss not improving.
            - `l2_reg` (float): L2 regularization parameter.
            - `loss_function` (str): Loss function, options are "mse", "mae", "binary_crossentropy", 
                "categorical_crossentropy", "kldiv", "nll".
            - `loss_function_params` (dict): Additional parameters for the loss function, if any.
            - `metrics` (list): List of metrics to be evaluated during training. Necessary for Keras only.
            - `checkpoint_path` (str): Path to the checkpoint file to be saved. ex: "./best_weights.h5".
                or "./checkpoint.{epoch:02d}-{val_loss:.2f}.h5"


        ### Returns

        It returns an object that corresponds with an ANN model.
        It has the following attributes:
        - `net`: The Keras.Models object created with the Keras Sequential API
        """
        # super(Keras_ANN, self).__init__()
        if not hparams: hparams = self.sample_hparams
        self.hparams = hparams
        self.history = None
        self.net = None
        self._layers_vec = []
        self._sizevec = []
        self._width = hparams.get("width")
        self._insize = hparams["input_size"]
        self._outsize = hparams["output_size"]
        self._dropout = hparams.get("dropout")
        self._depth = hparams.get("depth")
        self._denseactivation = actdict_keras[hparams["hidden_activation"]]
        self._outactivation = \
            actdict_keras[hparams.get("output_activation")] if hparams.get("output_activation") else None
        self._batchnorm = hparams.get("batchnorm")
        self._batchnormparams = hparams.get("batchnorm_params")
        self._learnrate = hparams.get("learning_rate")
        self._learnrate_decay_gamma = hparams.get("learning_rate_decay_gamma")
        self._optimizer = hparams.get("optimizer")
        self._optimizerparams = hparams.get("optimizer_params")
        self._loss = hparams.get("loss_function")
        self._metrics_list = hparams.get("metrics")
        self._epochs = hparams.get("epochs")
        self._batchsize = hparams.get("batch_size")
        self._l2_reg = hparams.get("l2_reg") if hparams.get("l2_reg") else 0.0
        self._callbacks = [MyCustomCallback()]
        self._earlystop = hparams.get("validation_tolerance_epochs")
        self._es = EarlyStopping(monitor='val_loss', mode="min", patience=self._earlystop) if self._earlystop else None
        if self._es:
            self._callbacks.append(self._es)
        self._chkpt = hparams.get("checkpoint_path")
        if self._chkpt:
            self._chk = ModelCheckpoint(self._chkpt, monitor='val_loss', verbose=0, save_best_only=True, mode='min')
        else:
            self._chk = None
        if self._chk:
            self._callbacks.append(self._chk)
        self.batch_input_shape = (self._batchsize, self._insize)
        self.batch_output_shape = (self._batchsize, self._outsize)
        self.early_stopping_monitor = hparams.get("early_stopping_monitor")
        self.early_stopping_mode = hparams.get("early_stopping_mode")
        self.early_stopping_value = hparams.get("early_stopping_value")
        if self.early_stopping_monitor:
            self._es_crit = EarlyStopAtCriteria(monitor=self.early_stopping_monitor, mode=self.early_stopping_mode,
                                                value=self.early_stopping_value)
            self._callbacks.append(self._es_crit)
        else:
            self._es_crit = None
        
        # Constructing the layer size vector (does not include input and output layers)
        if "list" in type(self._width).__name__.lower() or\
            "numpy" in type(self._width).__name__.lower():
            self._sizevec = self._width
        elif self._width == "auto":
            old = int(2**np.ceil(math.log2(self._insize)))
            for i in range(self._depth):
                new = int((2 if i < np.ceil(self._depth/2) else 0.5)*2**np.round(math.log2(old)))
                old = new
                self._sizevec.append(new)
        elif self._width is not None:
            self._sizevec = [self._width for _ in range(self._depth)]
        else:
            self._width = self._insize
            self._sizevec = [self._width for _ in range(self._depth)]

        # Constructing dense layers
        for i,size in enumerate(self._sizevec):
            if i==0:
                self._layers_vec.append(Dense(size, kernel_initializer=GLOROTUNIFORM, input_shape=(self._insize,),
                kernel_regularizer=regularizers.l2(self._l2_reg)))
            else:
                self._layers_vec.append(Dense(size, kernel_initializer=GLOROTUNIFORM,
                kernel_regularizer=regularizers.l2(self._l2_reg)))
            if self._batchnorm == "before":
                if self._batchnormparams:
                    self._layers_vec.append(BatchNormalization(**self._batchnormparams))
                else:
                    self._layers_vec.append(BatchNormalization())
            self._layers_vec.append(Activation(self._denseactivation))
            if self._batchnorm == "after":
                if self._batchnormparams:
                    self._layers_vec.append(BatchNormalization(**self._batchnormparams))
                else:
                    self._layers_vec.append(BatchNormalization())
            if self._dropout:
                self._layers_vec.append(Dropout(self._dropout))
            #old = size
        self._layers_vec.append(Dense(self._outsize, kernel_initializer=GLOROTUNIFORM,
                kernel_regularizer=regularizers.l2(self._l2_reg)))
        if self._outactivation:
            self._layers_vec.append(Activation(self._outactivation))

        #self.net = Sequential(self._layers_vec)
        # Use functional API
        self.input_layer = tf.keras.Input(shape=(self._insize,))
        x = self._layers_vec[0](self.input_layer)
        for layer in self._layers_vec[1:]:
            x = layer(x)
        self.output_layer = x
        self.net = tf.keras.Model(inputs=self.input_layer, outputs=self.output_layer)
        
        
        
        
    # def build(self):
    #     super().build(input_shape=self.batch_input_shape) 
        
    # def summary(self):
    #     x = Input(shape=(self._insize,))
    #     model = Model(inputs=[x], outputs=self.call(x))
    #     return model.summary()
        
    # def call(self, inputs):
    #     x = inputs
    #     for layer in self._layers_vec:
    #         x = layer(x)
    #     return x
        
    # def get_config(self):
    #     #return super().get_config().update(self.hparams)
    #     return self.hparams
    
    # @classmethod
    # def from_config(cls, hparams):
    #     return cls(hparams)
    
    
    def compile(self, num_samples=None):
        compile_keras_model(self.net, self._batchsize, self._learnrate, self._optimizer, self._loss, 
                              self._metrics_list, self._optimizerparams, self._learnrate_decay_gamma, num_samples)
        
        
    def fit(self, x_train, y_train, x_val, y_val, verbose:bool=True, **kwargs):
        self.history = fit_keras_model(self.net, x_train, y_train, x_val, y_val, 
            self._batchsize, self._epochs, self._callbacks, verbose, **kwargs)
        return self.history

    def __len__(self):
        return len(self._layers_vec)
    
    def __getitem__(self, key):
        return self._layers_vec[key]

    def __str__(self):
        s = \
            "ANN model with the following attributes:\n" + \
            "Input size:          " + str(self._insize)                             + "\n" + \
            "Output size:         " + str(self._outsize)                            + "\n" + \
            "Depth:               " + str(self._depth)                              + "\n" + \
            "Hidden layer widths: " + str(self._sizevec)                            + "\n" + \
            "Hidden activation:   " + str(self.hparams["hidden_activation"])        + "\n" + \
            "Output activation:   " + str(self.hparams.get("output_activation"))    + "\n" + \
            "Dropout rate:        " + str(self._dropout)                            + "\n" + \
            "Batch normalization: " + str(self._batchnorm)                          + "\n"
        return s

    def __repr__(self):
        return "ANN model with {} inputs, {} outputs, {} hidden layers of width {}.".format(
            self._insize, self._outsize, self._depth, self._sizevec)
    
    def train(self, x_train, x_val, y_train, y_val, verbose:bool=True, saveto:str=None, export:str=None, **kwargs):
        """Train the model according to its hyperparameters.

        ### Args:
            - `x_train` (numpy array): Training inputs
            - `x_val` (numpy array): Validation inputs
            - `y_train` (numpy array): Training target outputs
            - `y_val` (numpy array): Validation target outputs
            - `verbose` (bool, optional): Verbosity of training. Defaults to True.
            - `saveto` (str, optional): Save Keras model in path. Defaults to None.
            - `export` (str, optional): Save Keras model in .model file using keras2cpp for later use in C++. 
                Defaults to None.

        ### Returns:
            Nothing. It modifies the "net" attribute of the model, and the history of the training in self.history.
        
        """
        N = x_train.shape[0]
        self.compile(num_samples=N)
        _ = self.fit(x_train, y_train, x_val, y_val, verbose=verbose, **kwargs)
        if saveto:
            save_keras_model(self.net, self.history.history, saveto, self.hparams)
        if export:
            export_keras_model(self.net, export)


    def plot_history(self, metrics=['loss','val_loss'], fig_title='model loss', saveto:str=None, figsize=(10,5)):
        plot_keras_model_history(self.history.history, metrics, fig_title, saveto, figsize)
    







class Pytorch_ANN(nn.Module):
    
    sample_hparams = {
        "model_name": "Pytorch_ANN",
        "input_size": 10,
        "output_size": 3,
        "width": "auto",
        "depth": 2,
        "hidden_activation": "relu",
        "output_activation": None,
        "batchnorm": "before",
        "batchnorm_params": None,
        "dropout": 0.2,
        "learning_rate": 0.001,
        "learning_rate_decay_gamma": 0.99,
        "optimizer": "adam",
        "optimizer_params": {"eps": 1e-08},
        "batch_size": 32,
        "epochs": 2,
        "validation_tolerance_epochs": 2,
        "l2_reg": 0.0001,
        "loss_function": "categorical_crossentropy",
        "loss_function_params": None,
        "checkpoint_path": "dummy_Pytorch_ANN_best_weights.pt"
    }
    
    
    def __init__(self, hparams:dict=None):
        """Typical Artificial Neural Network class, also known as multilayer perceptron.
        This class will create a fully connected feedforward artificial neural network.
        It can be used for classification, regression, etc.
        It basically encompasses enough options to build all kinds of ANNs with any number of 
        inputs, outputs, layers with custom or arbitrary width or depth, etc.
        Supports multiple activation functions for hidden layers and the output layer,
        but the activation function of the hidden layers are all the same.
        
        ### Usage
        `net = Pytorch_ANN(hparams)` where `hparams` is the dictionary of hyperparameters.

        It can include the following keys:

            - `input_size` (int): number of inputs to the ANN, i.e. size of the input layer.
            - `output_size` (int): number of outputs to predict, i.e. size of the output layer.

            - `width` ("auto"|int|list|array): hidden layer width. "auto" decides automatically, 
                a number sets them all the same, and a list/array sets each hidden layer according to the list.
                If "auto", hidden layer widths will be set in such a way that the first half of the network will be the 
                encoder and the other half will be the decoder.
                Therefore, the first hidden layer will be twice as large as the input layer, 
                and every layer of the encoder will be twice as large as the previous one.
                In the decoder half, layer width will be halved until the output layer. 
                Layer widths will be powers of two.
            - `depth` (int): Specifies the depth of the network (number of hidden layers).
                It must be specified unless `width` is provided as a list. Then the depth will be inferred form it.
            
            - `hidden_activation` (str): Activation of the hidden layers.
                Supported activations are lowercase module names, e.g. "relu", "logsoftmax".
                LeakyReLU will have alpha=0.1.
            - `output_activation` (str): Activation of the output layer, if any.
                **Note**: For classification problems, you might want to choose "sigmoid", "softmax" or "logsoftmax".
                **Note**: For regression problems, no activation is needed. It is by default linear, 
                unless you want to manually specify an activation.
                **Note**: For classification problems, the output layer will be log-softmaxed when using PyTorch,
                if the loss function is chosen as crossentropy. 
                If the loss function is nll (negative loglikelihood) then you might want to specify the 
                output activation as logsoftmax or sigmoid.
            
            - `batchnorm` (str): If given, specifies where the batch normalization layer should be included: 
                `"before"` the activation, or `"after"` it.
                For activation functions such as **ReLU** and **sigmoid**, `"before"` is usually a better option. 
                For **tanh** and **LeakyReLU**, `"after"` is usually a better option.
            - `batchnorm_params` (dict): Dictionary of parameters for the batch normalization layer.
            
            - `dropout` (float): If given, specifies the dropout rate after every 
                hidden layer. It should be a probability value between 0 and 1.
            
            - `learning_rate` (float): Initial learning rate of training.
            - `learning_rate_decay_gamma` (float): Exponential decay rate gamma for learning rate, if any.
            
            - `optimizer` (str): Optimizer, options are "sgd" and "adam" for now.
            - `optimizer_params` (dict): Additional parameters of the optimizer, if any.
            
            - `batch_size` (int): Minibatch size for training.
            - `epochs` (int): Maximum number of epochs for training.
            - `validation_tolerance_epochs` (int): Epochs to tolerate unimproved val loss, before early stopping.
            - `l2_reg` (float): L2 regularization parameter.
            
            - `loss_function` (str): Loss function, options are "mse", "mae", "binary_crossentropy", 
                "categorical_crossentropy", "crossentropy", "kldiv", "nll".
            - `loss_function_params` (dict): Additional parameters for the loss function, if any.
            - `metrics` (list): List of metrics to be evaluated during training, similar to Keras
            - `checkpoint_path` (str): Path to the checkpoint file to be saved. ex: "./best_weights.h5".
                or "./checkpoint.{epoch:02d}-{val_loss:.2f}.h5"


        ### Returns

        It returns a `torch.nn.Module` object that corresponds with an ANN model.
        run `print(net)` afterwards to see what the ANN holds.

        """
        super(Pytorch_ANN, self).__init__()
        if not hparams: hparams = self.sample_hparams
        self.hparams = hparams
        self.layers = []
        self._sizevec = []
        self._insize = hparams["input_size"]
        self._outsize = hparams["output_size"]
        self._dropout = hparams.get("dropout")
        self._width = hparams.get("width")
        self._depth = hparams.get("depth")
        self._denseactivation = actdict_pytorch[hparams["hidden_activation"]]
        self._outactivation = \
            actdict_pytorch[hparams.get("output_activation")] if hparams.get("output_activation") else None
        self._batchnorm = hparams.get("batchnorm")
        self._batchnorm_params = hparams.get("batchnorm_params")
        self._learnrate = hparams.get("learning_rate")
        self._learnrate_decay_gamma = hparams.get("learning_rate_decay_gamma")
        self._optimizer = hparams.get("optimizer")
        self._optimizerparams = hparams.get("optimizer_params")
        self._lossfunctionparams = hparams.get("loss_function_params")
        self._loss = hparams.get("loss_function")
        self._metrics_list = hparams.get("metrics")
        self._epochs = hparams.get("epochs")
        self._batchsize = hparams.get("batch_size")
        self._l2_reg = hparams.get("l2_reg") if hparams.get("l2_reg") else 0.0
        self._earlystop = hparams.get("validation_tolerance_epochs")
        self._validation_data = hparams["validation_data"] if hparams.get("validation_data") else 0.1
        self.batch_input_shape = (self._batchsize, self._insize)
        self.batch_output_shape = (self._batchsize, self._outsize)
        self.history = None
        if self._l2_reg > 0.0:
            if self._optimizerparams is not None:
                self._optimizerparams["weight_decay"] = self._l2_reg
            else:
                self._optimizerparams = {"weight_decay": self._l2_reg}



        # Constructing the layer size vector (does not include input and output layers)
        if "list" in type(self._width).__name__ or "numpy" in type(self._width).__name__:
            self._sizevec = self._width
        elif self._width == "auto":
            old = int(2**np.ceil(math.log2(self._insize)))
            for i in range(self._depth):
                new = int((2 if i < np.ceil(self._depth/2) else 0.5)*2**np.round(math.log2(old)))
                old = new
                self._sizevec.append(new)
        elif self._width is not None:
            self._sizevec = [self._width for _ in range(self._depth)]
        else:
            self._width = self._insize
            self._sizevec = [self._width for _ in range(self._depth)]
        
        
        # Constructing layers
        old = self._insize
        new = self._sizevec[0]
        for width in self._sizevec:
            new = width
            l = nn.Linear(old, new)
            xavier_uniform_(l.weight)
            zeros_(l.bias)
            self.layers.append(l)
            if self._batchnorm == "before":
                if self._batchnorm_params:
                    self.layers.append(nn.BatchNorm1d(new, **self._batchnorm_params))
                else:
                    self.layers.append(nn.BatchNorm1d(new))
            self.layers.append(self._denseactivation)
            if self._batchnorm == "after":
                if self._batchnorm_params:
                    self.layers.append(nn.BatchNorm1d(new, **self._batchnorm_params))
                else:
                    self.layers.append(nn.BatchNorm1d(new))
            if self._dropout:
                self.layers.append(nn.Dropout(self._dropout))
            old = new
        self.layers.append(nn.Linear(old, self._outsize))
        if self._outactivation:
            self.layers.append(self._outactivation)
        
        # Sequentiating the layers
        self.net = nn.Sequential(*self.layers)


    def forward(self, x):
        return self.net(x)
    
    def __len__(self):
        return len(self.layers)
    
    def __getitem__(self, key):
        return self.layers[key]
    
    def train_model(self, dataset, verbose:bool=True, script_before_save:bool=False, saveto:str=None, **kwargs):
        self.history = train_pytorch_model(self, dataset, self._batchsize, self._loss, self._optimizer, 
            self._optimizerparams, self._lossfunctionparams, self._learnrate, 
            self._learnrate_decay_gamma, self._epochs, self._earlystop, self._validation_data, 
            verbose, script_before_save, saveto, **kwargs)
        
    def evaluate_model(self, dataset, verbose:bool=True, **kwargs):
        return evaluate_pytorch_model(self, dataset, loss_str=self._loss, loss_function_params=self._lossfunctionparams,
            batch_size=self._batchsize, device_str="cuda", verbose=verbose, **kwargs)
        
    def predict_model(self, dataset, 
        return_in_batches:bool=True, return_inputs:bool=False, return_raw_predictions:bool=False, 
        verbose:bool=True, **kwargs):
        return predict_pytorch_model(self, dataset, self._loss, self._batchsize, "cuda", 
            return_in_batches, return_inputs, return_raw_predictions, verbose, **kwargs)
        
        
        
        
