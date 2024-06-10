import os
import argparse
import random

import torch
import torch.nn as nn
import torch.optim as optim

from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from lra_config import config
from model import wrapper
from utils import lra_dataloader, early_stopping, opt, los, metrices
from tqdm import tqdm


def set_env(seed = 3407) -> None:
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)

def get_parameters(dataset_name):
    parser = argparse.ArgumentParser(description=f"Configure the parameters for {dataset_name} task.")
    task_config = config[dataset_name]

    for key, value in task_config.items():
        key_type = type(value)
        if key_type is bool:
            action = 'store_false' if value else 'store_true'
            parser.add_argument(f'--{key}', action=action, default=value, help=f'{key} (default: {value})')
        elif key_type in [int, float, str]:
            parser.add_argument(f'--{key}', type=key_type, default=value, help=f'{key} (default: {value})')
        else:
            raise ValueError(f"Unsupported type for key: {key}")

    args = parser.parse_args()
    print('Training configs: {}'.format(args))

    # Running in Nvidia GPU (CUDA) or CPU
    if args.enable_cuda and torch.cuda.is_available():
        # Set available CUDA devices
        # This option is crucial for multiple GPUs
        # 'cuda' ≡ 'cuda:0'
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    return args, device

def prepare_model(args, device):
    torch.autograd.set_detect_anomaly(True)

    model = wrapper.ConverterLRASingle(args).to(device)

    loss_nll = nn.NLLLoss()
    loss_seq_kp = los.KernelPolynomialLoss(batch_size = args.batch_size, max_order = args.max_order)
    loss_feat_kp = los.KernelPolynomialLoss(batch_size = args.batch_size, max_order = args.max_order)

    es = early_stopping.EarlyStopping(mode='min', min_delta=0.0, patience=args.patience)
    
    if args.optimizer == 'adamw':
        optimizer = optim.AdamW(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'lion':
        optimizer = opt.Lion(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'tiger':
        optimizer = opt.Tiger(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'sophia': # default optimizer
        optimizer = opt.SophiaG(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise ValueError(f'ERROR: The {args.optimizer} optimizer is undefined.')
    
    scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=3, eta_min=0.0005)

    return model, loss_nll, loss_seq_kp, loss_feat_kp, optimizer, scheduler, es

def prepare_data(args):
    assert args.dataset_name in ['image', 'text', 'listops', 'pathfinder', 'retrieval', 'path-x']

    if args.dataset_name == 'image':
        data_train = torch.load('./data/lra/image/image_train.pt').to(torch.int32)
        target_train = torch.load('./data/lra/image/image_train_target.pt').to(torch.int32)

        data_val = torch.load('./data/lra/image/image_test.pt').to(torch.int32)
        target_val = torch.load('./data/lra/image/image_test_target.pt').to(torch.int32)

        data_test = torch.load('./data/lra/image/image_test.pt').to(torch.int32)
        target_test = torch.load('./data/lra/image/image_test_target.pt').to(torch.int32)
    elif args.dataset_name == 'text':
        data_train = torch.load('./data/lra/text/text_train.pt').to(torch.int32)
        target_train = torch.load('./data/lra/text/text_train_target.pt').to(torch.int32)

        data_val = torch.load('./data/lra/text/text_test.pt').to(torch.int32)
        target_val = torch.load('./data/lra/text/text_test_target.pt').to(torch.int32)

        data_test = torch.load('./data/lra/text/text_test.pt').to(torch.int32)
        target_test = torch.load('./data/lra/text/text_test_target.pt').to(torch.int32)
    elif args.dataset_name == 'listops':
        data_train = torch.load('./data/lra/listops/listops_train.pt').to(torch.int32)
        target_train = torch.load('./data/lra/listops/listops_train_target.pt').to(torch.int32)

        data_val = torch.load('./data/lra/listops/listops_val.pt').to(torch.int32)
        target_val = torch.load('./data/lra/listops/listops_val_target.pt').to(torch.int32)

        data_test = torch.load('./data/lra/listops/listops_test.pt').to(torch.int32)
        target_test = torch.load('./data/lra/listops/listops_test_target.pt').to(torch.int32)
    elif args.dataset_name == 'pathfinder':
        data_train = torch.load('./data/lra/pathfinder/pathfinder_train.pt').to(torch.int32)
        target_train = torch.load('./data/lra/pathfinder/pathfinder_train_target.pt').to(torch.int32)

        data_val = torch.load('./data/lra/pathfinder/pathfinder_val.pt').to(torch.int32)
        target_val = torch.load('./data/lra/pathfinder/pathfinder_val_target.pt').to(torch.int32)

        data_test = torch.load('./data/lra/pathfinder/pathfinder_test.pt').to(torch.int32)
        target_test = torch.load('./data/lra/pathfinder/pathfinder_test_target.pt').to(torch.int32)
    elif args.dataset_name == 'retrieval':
        data_train = torch.load('./data/lra/retrieval/retrieval_train.pt').to(torch.int32)
        target_train = torch.load('./data/lra/retrieval/retrieval_train_target.pt').to(torch.int32)

        data_val = torch.load('./data/lra/retrieval/retrieval_val.pt').to(torch.int32)
        target_val = torch.load('./data/lra/retrieval/retrieval_val_target.pt').to(torch.int32)

        data_test = torch.load('./data/lra/retrieval/retrieval_test.pt').to(torch.int32)
        target_test = torch.load('./data/lra/retrieval/retrieval_test_target.pt').to(torch.int32)
    else:
        data_train = torch.load('./data/lra/path-x/path-x_train.pt').to(torch.int32)
        target_train = torch.load('./data/lra/path-x/path-x_train_target.pt').to(torch.int32)

        data_val = torch.load('./data/lra/path-x/path-x_val.pt').to(torch.int32)
        target_val = torch.load('./data/lra/path-x/path-x_val_target.pt').to(torch.int32)

        data_test = torch.load('./data/lra/path-x/path-x_test.pt').to(torch.int32)
        target_test = torch.load('./data/lra/path-x/path-x_test_target.pt').to(torch.int32)

    if args.pooling_type == 'CLS':
        cls_token_data_train = torch.tensor([[args.vocab_size - 1] * data_train.size(0)]).T
        cls_token_data_val = torch.tensor([[args.vocab_size - 1] * data_val.size(0)]).T
        cls_token_data_test = torch.tensor([[args.vocab_size - 1] * data_test.size(0)]).T

        data_train = torch.cat([cls_token_data_train, data_train], dim=-1)
        data_val = torch.cat([cls_token_data_val, data_val], dim=-1)
        data_test = torch.cat([cls_token_data_test, data_test], dim=-1)

    dataset_train = lra_dataloader.DatasetCreator(
        data = data_train,
        labels = target_train        
    )

    dataset_val = lra_dataloader.DatasetCreator(
        data = data_val,
        labels = target_val
    )

    dataset_test = lra_dataloader.DatasetCreator(
        data = data_test,
        labels = target_test
    )

    dataloader_train = DataLoader(
        dataset = dataset_train,
        batch_size = args.batch_size,
        shuffle = True,
        drop_last = True,
        num_workers = 1
    )

    dataloader_val = DataLoader(
        dataset = dataset_val,
        batch_size = args.batch_size,
        shuffle = False,
        drop_last = True,
        num_workers = 1
    )

    dataloader_test = DataLoader(
        dataset = dataset_test,
        batch_size = args.batch_size,
        shuffle = False,
        drop_last = True,
        num_workers = 1
    )

    return dataloader_train, dataloader_val, dataloader_test

def run(args, model, optimizer, scheduler, es, train_loader, val_loader, loss_nll, loss_seq_kp, loss_feat_kp, device):
    for _ in range(1, args.epochs + 1):
        acc_train, loss_train = train(model, optimizer, scheduler, train_loader, loss_nll, loss_seq_kp, loss_feat_kp, device)
        acc_val, loss_val = val(model, val_loader, loss_nll, loss_seq_kp, loss_feat_kp, device)
        print(f'train acc: {acc_train * 100: .2f}%')
        print(f'train loss: {loss_train: .2f}')
        print(f'val acc: {acc_val * 100: .2f}%')
        print(f'val loss: {loss_val: .2f}')

        if es.step(loss_val) and acc_val >= float(args.criteria):
            print('Early stopping.')
            break

    return loss_train, acc_train, loss_val, acc_val

def train(model, optimizer, scheduler, dataloader, loss_nll, loss_seq_kp, loss_feat_kp, device):
    model.train()

    acc_meter = metrices.AverageMeter()
    loss_meter = metrices.AverageMeter()

    for _, (samples, targets) in tqdm(enumerate(dataloader), total=len(dataloader)):
        samples = samples.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        preds = model(samples)
        acc_train = metrices.accuracy(preds.squeeze(), targets)
        loss_train = loss_nll(preds.squeeze(), targets)
        loss_train.backward()
        optimizer.step()
        # scheduler.step()

        acc_meter.update(acc_train.item(), targets.size(0))
        loss_meter.update(loss_train.item(), targets.size(0))

    return acc_meter.avg, loss_meter.avg

@torch.no_grad()
def val(model, dataloader, loss_nll, loss_seq_kp, loss_feat_kp, device):
    model.train()

    loss_meter = metrices.AverageMeter()
    acc_meter = metrices.AverageMeter()

    for _, (samples, targets) in tqdm(enumerate(dataloader), total=len(dataloader)):
        samples = samples.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        preds = model(samples)
        acc_train = metrices.accuracy(preds.squeeze(), targets)
        loss_train = loss_nll(preds.squeeze(), targets)

        acc_meter.update(acc_train.item(), targets.size(0))
        loss_meter.update(loss_train.item(), targets.size(0))

    return acc_meter.avg, loss_meter.avg

@torch.no_grad()
def test(model, dataloader, loss_nll, loss_seq_kp, loss_feat_kp, device):
    model.train()

    loss_meter = metrices.AverageMeter()
    acc_meter = metrices.AverageMeter()

    for _, (samples, targets) in tqdm(enumerate(dataloader), total=len(dataloader)):
        samples = samples.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        preds = model(samples)
        acc_train = metrices.accuracy(preds.squeeze(), targets)
        loss_train = loss_nll(preds.squeeze(), targets)

        acc_meter.update(acc_train.item(), targets.size(0))
        loss_meter.update(loss_train.item(), targets.size(0))

    return acc_meter.avg, loss_meter.avg

def calc_correct(out, label):
    preds = out.argmax(dim=-1)
    correct = preds.eq(label).sum().item()

    return correct

if __name__ == '__main__':
    SEED = 3407
    set_env(SEED)

    args, device = get_parameters("image")
    model, loss_nll, loss_seq_kp, loss_feat_kp, optimizer, scheduler, es = prepare_model(args, device)
    dataloader_train, dataloader_val, dataloader_test = prepare_data(args)
    loss_train, acc_train, loss_val, acc_val = run(args, model, optimizer, scheduler, es, dataloader_train, dataloader_val, loss_nll, loss_seq_kp, loss_feat_kp, device)
    loss_test, acc_test = test(model, dataloader_test, loss_nll, loss_seq_kp, loss_feat_kp, device)

    print(f'test acc: {acc_test * 100: .2f}%')
    print(f'test loss: {loss_test: .2f}')