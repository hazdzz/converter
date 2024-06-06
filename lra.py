import os
import argparse
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from torch.optim.lr_scheduler import CosineAnnealingLR
# from torchvision import transforms
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from lra_config import config
from model import wrapper
from utility import lra_dataloader, early_stopping, opt, los


def set_env(seed = 3407) -> None:
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
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

    # es = early_stopping.EarlyStopping(mode='min', min_delta=0.0, patience=args.patience)
    
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
    
    scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=3, eta_min=0.002)

    return model, loss_nll, loss_seq_kp, optimizer, scheduler

def prepare_data(args):
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
    # elif args.dataset_name == 'retrieval':
    #     data_train = torch.load('./data/lra/retrieval/retrieval_train.pt').to(torch.int32)
    #     target_train = torch.load('./data/lra/retrieval/retrieval_train_target.pt').to(torch.int32)

    #     data_val = torch.load('./data/lra/retrieval/retrieval_val.pt').to(torch.int32)
    #     target_val = torch.load('./data/lra/retrieval/retrieval_val_target.pt').to(torch.int32)

    #     data_test = torch.load('./data/lra/retrieval/retrieval_test.pt').to(torch.int32)
    #     target_test = torch.load('./data/lra/retrieval/retrieval_test_target.pt').to(torch.int32)
    # else:
    #     data_train = torch.load('./data/lra/path-x/path-x_train.pt').to(torch.int32)
    #     target_train = torch.load('./data/lra/path-x/path-x_train_target.pt').to(torch.int32)

    #     data_val = torch.load('./data/lra/path-x/path-x_val.pt').to(torch.int32)
    #     target_val = torch.load('./data/lra/path-x/path-x_val_target.pt').to(torch.int32)

    #     data_test = torch.load('./data/lra/path-x/path-x_test.pt').to(torch.int32)
    #     target_test = torch.load('./data/lra/path-x/path-x_test_target.pt').to(torch.int32)

    if args.pooling_type == 'CLS':
        cls_token_data_train = torch.tensor([[args.vocab_size - 1] * data_train.size(0)]).T
        cls_token_data_val = torch.tensor([[args.vocab_size - 1] * data_val.size(0)]).T
        cls_token_data_test = torch.tensor([[args.vocab_size - 1] * data_test.size(0)]).T

        data_train = torch.cat([cls_token_data_train, data_train], -1)
        data_val = torch.cat([cls_token_data_val, data_val], -1)
        data_test = torch.cat([cls_token_data_test, data_test], -1)

    train_set = lra_dataloader.DatasetCreator(
        data = data_train,
        labels = target_train        
    )

    train_loader = DataLoader(
        dataset = train_set,
        batch_size = args.batch_size,
        shuffle = True,
        drop_last = True,
        num_workers = 1
    )

    val_set = lra_dataloader.DatasetCreator(
        data = data_val,
        labels = target_val
    )

    val_loader = DataLoader(
        dataset = val_set,
        batch_size = args.batch_size,
        shuffle = False,
        drop_last = True,
        num_workers = 1
    )

    test_set = lra_dataloader.DatasetCreator(
        data = data_test,
        labels = target_test
    )

    test_loader = DataLoader(
        dataset = test_set,
        batch_size = args.batch_size,
        shuffle = False,
        drop_last = True,
        num_workers = 1
    )

    return train_loader, val_loader, test_loader

def run(args, model, optimizer, scheduler, train_loader, val_loader, loss_nll, loss_seq_kp, device):
    for _ in range(1, args.epochs + 1):
        loss_train, acc_train = train(model, optimizer, scheduler, train_loader, loss_nll, loss_seq_kp, device)
        loss_val, acc_val = val(model, val_loader, loss_nll, loss_seq_kp, device)
        print(f'train acc: {acc_train * 100: .2f}%')
        print(f'train loss: {loss_train: .2f}')
        print(f'val acc: {acc_val * 100: .2f}%')
        print(f'val loss: {loss_val: .2f}')

    return loss_train, acc_train, loss_val, acc_val

def train(model, optimizer, scheduler, train_loader, loss_nll, loss_seq_kp, device):
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    with tqdm(train_loader, unit="batch") as ttrain:
        ttrain.set_description("Training")

        for batch in ttrain:
            data_train, target_train = batch
            data_train, target_train = data_train.to(device), target_train.to(device)
            
            optimizer.zero_grad()
            train_out = model(data_train)
            loss_train = loss_nll(train_out.squeeze(), target_train)
            # loss_train = loss_nll(train_out.squeeze(), target_train) + loss_seq_kp(model.converter.chsyconv.seq_kernel_poly.gibbs_damp)
            loss_train.backward()
            optimizer.step()
            # scheduler.step()

            total_loss += loss_train.item() * data_train.size(0)
            total_correct += calc_correct(train_out, target_train)
            total_samples += data_train.size(0)

            ttrain.set_postfix(loss=loss_train.item())

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    return avg_loss, accuracy

@torch.no_grad()
def val(model, val_loader, loss_nll, loss_seq_kp, device):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    with tqdm(val_loader, unit="batch") as tval:
        tval.set_description("Validation")

        for batch in tval:
            data_val, target_val = batch
            data_val, target_val = data_val.to(device), target_val.to(device)

            val_out = model(data_val)
            loss_val = loss_nll(val_out.squeeze(), target_val)
            # loss_val = loss_nll(val_out.squeeze(), target_val) + loss_seq_kp(model.converter.chsyconv.seq_kernel_poly.gibbs_damp)
            total_loss += loss_val.item() * data_val.size(0)
            total_correct += calc_correct(val_out, target_val)
            total_samples += data_val.size(0)

            tval.set_postfix(loss=loss_val.item())

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    
    return avg_loss, accuracy

@torch.no_grad()
def test(model, test_loader, loss_nll, loss_seq_kp, device):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    with tqdm(test_loader, unit="batch") as ttest:
        ttest.set_description("Testing")

        for batch in ttest:
            data_test, target_test = batch
            data_test, target_test = data_test.to(device), target_test.to(device)

            test_out = model(data_test)
            loss_test = loss_nll(test_out.squeeze(), target_test)
            # loss_test = loss_nll(test_out.squeeze(), target_test) + loss_seq_kp(model.converter.chsyconv.seq_kernel_poly.gibbs_damp)
            total_loss += loss_test.item() * data_test.size(0)
            total_correct += calc_correct(test_out, target_test)
            total_samples += data_test.size(0)

            ttest.set_postfix(loss=loss_test.item())

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    return avg_loss, accuracy

def calc_correct(out, label):
    preds = out.argmax(dim=-1)
    correct = preds.eq(label).sum().item()

    return correct

if __name__ == '__main__':
    SEED = 3407
    set_env(SEED)

    args, device = get_parameters("text")
    model, loss_nll, loss_seq_kp, optimizer, scheduler = prepare_model(args, device)
    train_loader, val_loader, test_loader = prepare_data(args)
    loss_train, acc_train, loss_val, acc_val = run(args, model, optimizer, scheduler, train_loader, val_loader, loss_nll, loss_seq_kp, device)
    loss_test, acc_test = test(model, test_loader, loss_nll, loss_seq_kp, device)

    print(f'test acc: {acc_test * 100: .2f}%')
    print(f'test loss: {loss_test: .2f}')