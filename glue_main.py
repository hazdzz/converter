import os
import time
import random
import numpy as np
import pandas as pd
import argparse
import logging as log
import _pickle as pkl
import ipdb as pdb
from sklearn.metrics import f1_score, accuracy_score

import torch
import torch.nn as nn
import torch.optim as optim

from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from model.converter import wrapper
from utils import glue_dataloader, early_stopping, opt, los, metrices

def set_env(seed = 3407) -> None:
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def prepare_model(args, device):
    torch.autograd.set_detect_anomaly(True)

    if args.dataset == 'retrieval':
        model = wrapper.LRADual(args).to(device)
    else:
        model = wrapper.LRASingle(args).to(device)

    loss_nll = nn.NLLLoss()
    loss_bcewl = nn.BCEWithLogitsLoss()
    loss_seq_kp = los.KernelPolynomialLoss(batch_size=args.batch_size, max_order=args.max_order, eta=args.eta)

    es = early_stopping.EarlyStopping(delta=0.0, 
                                      patience=args.patience,
                                      verbose=True, 
                                      path="converter_" + args.dataset + ".pt")
    
    if args.optimizer == 'adamw': 
        optimizer = optim.AdamW(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'nadamw': # default
        optimizer = optim.NAdam(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay, decoupled_weight_decay=True)
    elif args.optimizer == 'adan':
        optimizer = opt.Adan(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'lion':
        optimizer = opt.Lion(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'sophia':
        optimizer = opt.SophiaG(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise ValueError(f'ERROR: The {args.optimizer} optimizer is undefined.')
    
    scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=3, eta_min=0.0005)

    return model, loss_nll, loss_seq_kp, optimizer, scheduler, es


def train(model, iterator, optimizer, criterion, metric, device):
    epoch_loss = 0
    epoch_acc = 0
    model.train()

    for batch in iterator:
        optimizer.zero_grad()
        
        try:
            text, label = batch.text, batch.label
        except:
            batch = tuple(t.to(device) for t in batch)
            text, label = batch

        pred = model(text).squeeze(1)
        loss = criterion(pred, label)
        acc = metric(pred, label)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def eval(model, iterator, criterion, metric, device):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()

    epoch_label, epoch_pred = [], []
    
    with torch.no_grad():
        for batch in iterator:
            try:
                text, label = batch.text, batch.label
            except:
                batch = tuple(t.to(device) for t in batch)
                text, label = batch

            pred = model(text).squeeze(1)
            loss = criterion(pred, label)
            acc = metric(pred, label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

            epoch_label.extend(label.tolist())
            epoch_pred.extend(pred.tolist())

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))

    return elapsed_mins, elapsed_secs


def binary_accuracy(preds, y):
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()
    acc = correct.sum() / len(correct)

    return acc


def f1_torch(preds, y):
    y_pred_new = torch.argmax(torch.sigmoid(preds),dim=1).to('cpu').numpy()
    y_true_new = y.to('cpu').numpy()
    f1_res = f1_score(y_true_new, y_pred_new, average='micro')

    return f1_res


def accuracy_torch(preds, y):
    y_pred_new = torch.argmax(torch.sigmoid(preds),dim=1).to('cpu').numpy()
    y_true_new = y.to('cpu').numpy()
    acc = accuracy_score(y_true_new, y_pred_new)

    return acc


def matthews_corr(preds, y):
    y_pr_torch = torch.round(torch.sigmoid(preds))
    y_tr_torch = torch.round(y)

    tp = ((y_tr_torch==1) & (y_pr_torch==1)).sum().to('cpu').numpy()
    tn = ((y_tr_torch==0) & (y_pr_torch==0)).sum().to('cpu').numpy()
    fp = ((y_tr_torch==0) & (y_pr_torch==1)).sum().to('cpu').numpy()
    fn = ((y_tr_torch==1) & (y_pr_torch==0)).sum().to('cpu').numpy()

    try:
        mcc = (tp * tn - fp * fn) / np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    except:
        mcc=1

    return mcc


def train_model(trn_iter, val_iter, model, optimizer, criterion, metric, exp_name, task_type, device, N_EPOCHS = 12):
    print('=====================')
    print(f'Running experiment {exp_name}')
    best_valid_loss = float('inf')
    model_res=[]
    log_col=['Exp Name', 'Epoch', 'Epoch Mins', 'Epoch Secs', 'Train Loss', 'Train Accuracy', 'Valid Loss', 'Valid Accuracy']
 
    for epoch in range(N_EPOCHS):
        start_time = time.time()
      
        train_loss, train_acc = train(model, trn_iter, optimizer, criterion, metric, device)

        valid_loss, valid_acc = eval(model, val_iter, criterion, metric, device)
          
        end_time = time.time()
          
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
          
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), exp_name + '-model.pt')
      
        print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')
        new_val = [exp_name, epoch, epoch_mins, epoch_secs, train_loss, train_acc, valid_loss, valid_acc, ]
        model_res = model_res+[new_val]
    
    df = pd.DataFrame(model_res, columns=log_col)

    return df