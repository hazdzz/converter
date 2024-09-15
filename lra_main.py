import os
import gc
import random
import argparse
import yaml
import warnings

import torch
import torch.nn as nn
import torch.optim as optim

from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.converter import wrapper
from utils import lra_dataloader, early_stopping, opt, los, metrices


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


def get_parameters():
    parser = argparse.ArgumentParser(description='Converter for lra data')
    parser.add_argument('--config', type=str, default='lra_config.yaml', help='Path to the yaml configuration file')
    parser.add_argument('--task', type=str, default='image', choices=['listops', 'image', 'pathfinder', 'text', 'retrieval'], help='Name of the task')
    args = parser.parse_args()

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    task_config = config[args.task]

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
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    else:
        device = torch.device('cpu')
        gc.collect()

    return args, device


def prepare_model(args, device):
    torch.autograd.set_detect_anomaly(True)

    if args.dataset == 'retrieval':
        model = wrapper.LRADual(args).to(device)
    else:
        model = wrapper.LRASingle(args).to(device)

    loss_cel = nn.CrossEntropyLoss()
    loss_seq_kp = los.KernelPolynomialLoss(batch_size=args.batch_size, max_order=args.max_order, eta=args.eta)

    es = early_stopping.EarlyStopping(delta=0.0, 
                                      patience=args.patience,
                                      verbose=True, 
                                      path="converter_" + args.dataset + ".pt")
    
    if args.optimizer == 'adamw': 
        optimizer = optim.AdamW(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'nadamw': # default
        optimizer = optim.NAdam(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay, decoupled_weight_decay=True)
    elif args.optimizer == 'ademamix':
        optimizer = opt.AdEMAMix(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise ValueError(f'ERROR: The {args.optimizer} optimizer is undefined.')
    
    scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=3, eta_min=0.0005)

    return model, loss_cel, loss_seq_kp, optimizer, scheduler, es


def prepare_data(args):
    assert args.dataset in ['image', 'text', 'listops', 'pathfinder', 'path-x']

    if args.dataset == 'image':
        data_train = torch.load('./data/lra/image/image_train.pt').to(torch.int32)
        target_train = torch.load('./data/lra/image/image_train_target.pt').to(torch.int32)

        data_val = torch.load('./data/lra/image/image_test.pt').to(torch.int32)
        target_val = torch.load('./data/lra/image/image_test_target.pt').to(torch.int32)

        data_test = torch.load('./data/lra/image/image_test.pt').to(torch.int32)
        target_test = torch.load('./data/lra/image/image_test_target.pt').to(torch.int32)
    elif args.dataset == 'text':
        data_train = torch.load('./data/lra/text/text_train.pt').to(torch.int32)
        target_train = torch.load('./data/lra/text/text_train_target.pt').to(torch.int32)

        data_val = torch.load('./data/lra/text/text_test.pt').to(torch.int32)
        target_val = torch.load('./data/lra/text/text_test_target.pt').to(torch.int32)

        data_test = torch.load('./data/lra/text/text_test.pt').to(torch.int32)
        target_test = torch.load('./data/lra/text/text_test_target.pt').to(torch.int32)
    elif args.dataset == 'listops':
        data_train = torch.load('./data/lra/listops/listops_train.pt').to(torch.int32)
        target_train = torch.load('./data/lra/listops/listops_train_target.pt').to(torch.int32)

        data_val = torch.load('./data/lra/listops/listops_val.pt').to(torch.int32)
        target_val = torch.load('./data/lra/listops/listops_val_target.pt').to(torch.int32)

        data_test = torch.load('./data/lra/listops/listops_test.pt').to(torch.int32)
        target_test = torch.load('./data/lra/listops/listops_test_target.pt').to(torch.int32)
    elif args.dataset == 'pathfinder':
        data_train = torch.load('./data/lra/pathfinder/pathfinder_train.pt').to(torch.int32)
        target_train = torch.load('./data/lra/pathfinder/pathfinder_train_target.pt').to(torch.int32)

        data_val = torch.load('./data/lra/pathfinder/pathfinder_val.pt').to(torch.int32)
        target_val = torch.load('./data/lra/pathfinder/pathfinder_val_target.pt').to(torch.int32)

        data_test = torch.load('./data/lra/pathfinder/pathfinder_test.pt').to(torch.int32)
        target_test = torch.load('./data/lra/pathfinder/pathfinder_test_target.pt').to(torch.int32)
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

    dataset_train = lra_dataloader.SingleDatasetCreator(
        data = data_train,
        labels = target_train        
    )

    dataset_val = lra_dataloader.SingleDatasetCreator(
        data = data_val,
        labels = target_val
    )

    dataset_test = lra_dataloader.SingleDatasetCreator(
        data = data_test,
        labels = target_test
    )

    dataloader_train = DataLoader(
        dataset = dataset_train,
        batch_size = args.batch_size,
        shuffle = True,
        drop_last = True,
        num_workers = args.num_workers
    )

    dataloader_val = DataLoader(
        dataset = dataset_val,
        batch_size = args.batch_size,
        shuffle = False,
        drop_last = True,
        num_workers = args.num_workers
    )

    dataloader_test = DataLoader(
        dataset = dataset_test,
        batch_size = args.batch_size,
        shuffle = False,
        drop_last = True,
        num_workers = args.num_workers
    )

    return dataloader_train, dataloader_val, dataloader_test


def run(args, model, optimizer, scheduler, es, train_loader, val_loader, loss_cel, loss_seq_kp, device):
    for _ in range(1, args.epochs + 1):
        acc_train, loss_train, peak_memory_train = train(args, model, optimizer, scheduler, train_loader, loss_cel, loss_seq_kp, device)
        acc_val, loss_val = val(args, model, val_loader, loss_cel, loss_seq_kp, device)
        print(f'train acc: {acc_train: .2f}%')
        print(f'train loss: {loss_train: .4f}')
        print(f'val acc: {acc_val: .2f}%')
        print(f'val loss: {loss_val: .4f}')

        es(loss_val, model)
        if es.early_stop:
            print("Early stopping")
            break
    
    return acc_train, loss_train, acc_val, loss_val, peak_memory_train


def train(args, model, optimizer, scheduler, dataloader, loss_cel, loss_seq_kp, device):
    model.train()

    acc_meter = metrices.AverageMeter()
    loss_meter = metrices.AverageMeter()

    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Training")

    for _, (samples, targets) in pbar:
        samples = samples.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        preds = model(samples)
        acc = torch.tensor(metrices.accuracy(preds.squeeze(), targets))
        loss = loss_cel(preds.squeeze(), targets)
        if (args.enable_kpm is True) and \
            (args.enable_kploss is True) and \
            (args.kernel_type == 'none' or args.kernel_type == 'dirichlet'):
            loss = loss + loss_seq_kp(model.converter.chsyconv.seq_kernel_poly.cheb_coef)
        loss.backward()
        optimizer.step()

        acc_meter.update(acc.item(), targets.size(0))
        loss_meter.update(loss.item(), targets.size(0))

        pbar.set_postfix(loss=loss_meter.avg)

    # scheduler.step()
    peak_memory = torch.cuda.max_memory_allocated()

    return acc_meter.avg, loss_meter.avg, peak_memory


@torch.no_grad()
def val(args, model, dataloader, loss_cel, loss_seq_kp, device):
    model.eval()

    loss_meter = metrices.AverageMeter()
    acc_meter = metrices.AverageMeter()

    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Validation")

    for _, (samples, targets) in pbar:
        samples = samples.to(device)
        targets = targets.to(device)

        preds = model(samples)
        acc = torch.tensor(metrices.accuracy(preds.squeeze(), targets))
        loss = loss_cel(preds.squeeze(), targets)
        if (args.enable_kpm is True) and \
            (args.enable_kploss is True) and \
            (args.kernel_type == 'none' or args.kernel_type == 'dirichlet'):
            loss = loss + loss_seq_kp(model.converter.chsyconv.seq_kernel_poly.cheb_coef)
        acc_meter.update(acc.item(), targets.size(0))
        loss_meter.update(loss.item(), targets.size(0))

        pbar.set_postfix(loss=loss_meter.avg)

    return acc_meter.avg, loss_meter.avg


@torch.no_grad()
def test(args, model, dataloader, loss_cel, loss_seq_kp, device):
    model.load_state_dict(torch.load("converter_" + args.dataset + ".pt"))
    model.eval()

    loss_meter = metrices.AverageMeter()
    acc_meter = metrices.AverageMeter()

    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Testing")

    for _, (samples, targets) in pbar:
        samples = samples.to(device)
        targets = targets.to(device)

        preds = model(samples)
        acc = torch.tensor(metrices.accuracy(preds.squeeze(), targets))
        loss = loss_cel(preds.squeeze(), targets)
        if (args.enable_kpm is True) and \
            (args.enable_kploss is True) and \
            (args.kernel_type == 'none' or args.kernel_type == 'dirichlet'):
            loss = loss + loss_seq_kp(model.converter.chsyconv.seq_kernel_poly.cheb_coef)
        acc_meter.update(acc.item(), targets.size(0))
        loss_meter.update(loss.item(), targets.size(0))

        pbar.set_postfix(loss=loss_meter.avg)

    return acc_meter.avg, loss_meter.avg


def prepare_data_retrieval(args):
    data_train_1 = torch.load('./data/lra/retrieval/retrieval_train_1.pt').to(torch.int32)
    data_train_2 = torch.load('./data/lra/retrieval/retrieval_train_2.pt').to(torch.int32)
    target_train = torch.load('./data/lra/retrieval/retrieval_train_target.pt').to(torch.int32)

    data_val_1 = torch.load('./data/lra/retrieval/retrieval_val_1.pt').to(torch.int32)
    data_val_2 = torch.load('./data/lra/retrieval/retrieval_val_2.pt').to(torch.int32)
    target_val = torch.load('./data/lra/retrieval/retrieval_val_target.pt').to(torch.int32)

    data_test_1 = torch.load('./data/lra/retrieval/retrieval_test_1.pt').to(torch.int32)
    data_test_2 = torch.load('./data/lra/retrieval/retrieval_test_2.pt').to(torch.int32)
    target_test = torch.load('./data/lra/retrieval/retrieval_test_target.pt').to(torch.int32)

    if args.pooling_type == 'CLS':
        cls_token_data_train_1 = torch.tensor([[args.vocab_size - 1] * data_train_1.size(0)]).T
        cls_token_data_val_1 = torch.tensor([[args.vocab_size - 1] * data_val_1.size(0)]).T
        cls_token_data_test_1 = torch.tensor([[args.vocab_size - 1] * data_test_1.size(0)]).T

        cls_token_data_train_2 = torch.tensor([[args.vocab_size - 1] * data_train_2.size(0)]).T
        cls_token_data_val_2 = torch.tensor([[args.vocab_size - 1] * data_val_2.size(0)]).T
        cls_token_data_test_2 = torch.tensor([[args.vocab_size - 1] * data_test_2.size(0)]).T

        data_train_1 = torch.cat([cls_token_data_train_1, data_train_1], dim=-1)
        data_val_1 = torch.cat([cls_token_data_val_1, data_val_1], dim=-1)
        data_test_1 = torch.cat([cls_token_data_test_1, data_test_1], dim=-1)

        data_train_2 = torch.cat([cls_token_data_train_2, data_train_2], dim=-1)
        data_val_2 = torch.cat([cls_token_data_val_2, data_val_2], dim=-1)
        data_test_2 = torch.cat([cls_token_data_test_2, data_test_2], dim=-1)

    dataset_train = lra_dataloader.DualDatasetCreator(
        data1 = data_train_1,
        data2 = data_train_2,
        labels = target_train        
    )

    dataset_val = lra_dataloader.DualDatasetCreator(
        data1 = data_val_1,
        data2 = data_val_2,
        labels = target_val
    )

    dataset_test = lra_dataloader.DualDatasetCreator(
        data1 = data_test_1,
        data2 = data_test_2,
        labels = target_test
    )

    dataloader_train = DataLoader(
        dataset = dataset_train,
        batch_size = args.batch_size,
        shuffle = True,
        drop_last = True,
        num_workers = args.num_workers
    )

    dataloader_val = DataLoader(
        dataset = dataset_val,
        batch_size = args.batch_size,
        shuffle = False,
        drop_last = True,
        num_workers = args.num_workers
    )

    dataloader_test = DataLoader(
        dataset = dataset_test,
        batch_size = args.batch_size,
        shuffle = False,
        drop_last = True,
        num_workers = args.num_workers
    )

    return dataloader_train, dataloader_val, dataloader_test


def run_retrieval(args, model, optimizer, scheduler, es, train_loader, val_loader, loss_cel, loss_seq_kp, device):
    for _ in range(1, args.epochs + 1):
        acc_train, loss_train, peak_memory_train = train_retrieval(args, model, optimizer, scheduler, train_loader, loss_cel, loss_seq_kp, device)
        acc_val, loss_val = val_retrieval(args, model, val_loader, loss_cel, loss_seq_kp, device)
        print(f'train acc: {acc_train: .2f}%')
        print(f'train loss: {loss_train: .4f}')
        print(f'val acc: {acc_val: .2f}%')
        print(f'val loss: {loss_val: .4f}')

        es(loss_val, model)
        if es.early_stop:
            print("Early stopping")
            break

    return acc_train, loss_train, acc_val, loss_val, peak_memory_train


def train_retrieval(args, model, optimizer, scheduler, dataloader, loss_cel, loss_seq_kp, device):
    model.train()

    acc_meter = metrices.AverageMeter()
    loss_meter = metrices.AverageMeter()

    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Training")

    for _, (samples_1, samples_2, targets) in pbar:
        samples_1 = samples_1.to(device)
        samples_2 = samples_2.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        preds = model(samples_1, samples_2)
        acc = torch.tensor(metrices.accuracy(preds.squeeze(), targets))
        loss = loss_cel(preds.squeeze(), targets)
        if (args.enable_kpm is True) and \
            (args.enable_kploss is True) and \
            (args.kernel_type == 'none' or args.kernel_type == 'dirichlet'):
            loss = loss + loss_seq_kp(model.converter.chsyconv.seq_kernel_poly.cheb_coef) 
        loss.backward()
        optimizer.step()

        acc_meter.update(acc.item(), targets.size(0))
        loss_meter.update(loss.item(), targets.size(0))

        pbar.set_postfix(loss=loss_meter.avg)

    # scheduler.step()
    peak_memory = torch.cuda.max_memory_allocated()

    return acc_meter.avg, loss_meter.avg, peak_memory


@torch.no_grad()
def val_retrieval(args, model, dataloader, loss_cel, loss_seq_kp, device):
    model.eval()

    loss_meter = metrices.AverageMeter()
    acc_meter = metrices.AverageMeter()

    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Validation")

    for _, (samples_1, samples_2, targets) in pbar:
        samples_1 = samples_1.to(device)
        samples_2 = samples_2.to(device)
        targets = targets.to(device)

        preds = model(samples_1, samples_2)
        acc = torch.tensor(metrices.accuracy(preds.squeeze(), targets))
        loss = loss_cel(preds.squeeze(), targets)
        if (args.enable_kpm is True) and \
            (args.enable_kploss is True) and \
            (args.kernel_type == 'none' or args.kernel_type == 'dirichlet'):
            loss = loss + loss_seq_kp(model.converter.chsyconv.seq_kernel_poly.cheb_coef) 

        acc_meter.update(acc.item(), targets.size(0))
        loss_meter.update(loss.item(), targets.size(0))

        pbar.set_postfix(loss=loss_meter.avg)

    return acc_meter.avg, loss_meter.avg


@torch.no_grad()
def test_retrieval(args, model, dataloader, loss_cel, loss_seq_kp, device):
    model.load_state_dict(torch.load("converter_" + args.dataset + ".pt"))
    model.eval()

    loss_meter = metrices.AverageMeter()
    acc_meter = metrices.AverageMeter()

    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Testing")

    for _, (samples_1, samples_2, targets) in pbar:
        samples_1 = samples_1.to(device)
        samples_2 = samples_2.to(device)
        targets = targets.to(device)

        preds = model(samples_1, samples_2)
        acc = torch.tensor(metrices.accuracy(preds.squeeze(), targets))
        loss = loss_cel(preds.squeeze(), targets)
        if (args.enable_kpm is True) and \
            (args.enable_kploss is True) and \
            (args.kernel_type == 'none' or args.kernel_type == 'dirichlet'):
            loss = loss + loss_seq_kp(model.converter.chsyconv.seq_kernel_poly.cheb_coef) 

        acc_meter.update(acc.item(), targets.size(0))
        loss_meter.update(loss.item(), targets.size(0))

        pbar.set_postfix(loss=loss_meter.avg)

    return acc_meter.avg, loss_meter.avg


if __name__ == '__main__':
    SEED = 3407
    set_env(SEED)

    warnings.filterwarnings("ignore", category=UserWarning)

    args, device = get_parameters()
    model, loss_cel, loss_seq_kp, optimizer, scheduler, es = prepare_model(args, device)
    if args.dataset == 'retrieval':
        dataloader_train, dataloader_val, dataloader_test = prepare_data_retrieval(args)
        acc_train, loss_train, acc_val, loss_val, peak_memory_train = run_retrieval(args, 
                                                                 model, 
                                                                 optimizer, 
                                                                 scheduler, 
                                                                 es, 
                                                                 dataloader_train, 
                                                                 dataloader_val,  
                                                                 loss_cel, 
                                                                 loss_seq_kp, 
                                                                 device)
        acc_test, loss_test = test_retrieval(args, model, dataloader_test, 
                                             loss_cel, loss_seq_kp, 
                                             device)
    else:
        dataloader_train, dataloader_val, dataloader_test = prepare_data(args)
        acc_train, loss_train, acc_val, loss_val, peak_memory_train = run(args, model, 
                                                       optimizer, scheduler, 
                                                       es, dataloader_train, 
                                                       dataloader_val, loss_cel, 
                                                       loss_seq_kp, device)
        acc_test, loss_test = test(args, model, dataloader_test, loss_cel, 
                                   loss_seq_kp, device)

    print(f'test acc: {acc_test: .2f}%')
    print(f'test loss: {loss_test: .4f}')
    print(f"Peak memory usage in traing: {peak_memory_train / (1024 ** 3):.2f} GiB")
