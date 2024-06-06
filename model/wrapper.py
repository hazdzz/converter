import torch
import torch.nn as nn
import torch.nn.init as init
from .norm import ScaleNorm, RMSNorm
from model import converter
from torch import Tensor


class SingleClassifier(nn.Module):
    def __init__(self, pooling_type, max_seq_len, encoder_dim, mlp_dim, num_class) -> None:
        super(SingleClassifier, self).__init__()
        self.pooling_type = pooling_type
        self.max_seq_len = max_seq_len
        self.encoder_dim = encoder_dim
        self.mlp_dim = mlp_dim
        self.num_class = num_class
        self.linear1 = nn.Linear(encoder_dim, mlp_dim)
        self.flatten_linear1 = nn.Linear(max_seq_len * encoder_dim, mlp_dim)
        self.linear2 = nn.Linear(mlp_dim, num_class)
        self.final_linear = nn.Linear(encoder_dim, num_class)
        self.flatten_final_linear = nn.Linear(max_seq_len * encoder_dim, num_class)
        self.pool_norm = ScaleNorm(mlp_dim)
        self.relu = nn.ReLU()
        self.gelu = nn.GELU()
        self.logsoftmax = nn.LogSoftmax(dim=-1)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.normal_(self.linear1.weight, mean=0, std=1 / self.encoder_dim)
        init.normal_(self.flatten_linear1.weight, mean=0, std=1 / (self.max_seq_len * self.encoder_dim))
        init.normal_(self.linear2.weight, mean=0, std=1 / self.mlp_dim)

        init.normal_(self.final_linear.weight, mean=0, std=1 / self.encoder_dim)
        init.normal_(self.flatten_final_linear.weight, mean=0, std=1 / (self.max_seq_len * self.encoder_dim))

    def pooling(self, input, mode):
        if mode == 'CLS':
            pooled = input[:, 0, :]
        elif mode == 'MEAN':
            pooled = input.mean(dim=1)
        elif mode == 'SUM':
            pooled = input.sum(dim=1)
        elif mode == 'FLATTEN':
            pooled = input.view(input.shape[0], -1)
        else:
            raise NotImplementedError('Pooling type is not supported.')
        
        return pooled

    def forward(self, encoded) -> Tensor:
        pooled = self.pooling(encoded, self.pooling_type)

        if self.pooling_type == 'FLATTEN':
            pooled1 = self.flatten_linear1(pooled)
        else:
            pooled1 = self.linear1(pooled)
        # pooled1 = self.relu(pooled1)
        pooled1 = self.gelu(pooled1)
        pooled1_norm = self.pool_norm(pooled1)
        pooled2 = self.linear2(pooled1_norm)
        classified = self.logsoftmax(pooled2)

        # if self.pooling_type == 'FLATTEN':
        #     pooled_linear = self.flatten_final_linear(pooled)
        # else:
        #     pooled_linear = self.final_linear(pooled)
        # classified = self.logsoftmax(pooled_linear)

        return classified

class DualClassifier(nn.Module):
    def __init__(self, pooling_type, max_seq_len, encoder_dim, mlp_dim, num_class, 
                 interaction) -> None:
        super(DualClassifier, self).__init__()
        self.pooling_type = pooling_type
        self.interaction = interaction
        self.linear1 = nn.Linear(encoder_dim * 2, mlp_dim)
        self.nli_linear1 = nn.Linear(encoder_dim * 4, mlp_dim)
        self.flatten_linear1 = nn.Linear(max_seq_len * encoder_dim * 2, mlp_dim)
        self.flatten_nli_linear1 = nn.Linear(max_seq_len * encoder_dim * 4, mlp_dim)
        self.linear2 = nn.Linear(mlp_dim, num_class)
        self.relu = nn.ReLU()
        self.gelu = nn.GELU()
        self.logsoftmax = nn.LogSoftmax(dim=-1)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_normal_(self.linear1.weight, nonlinearity='relu')
        init.kaiming_normal_(self.nli_linear1.weight, nonlinearity='relu')
        init.kaiming_normal_(self.flatten_linear1.weight, nonlinearity='relu')
        init.kaiming_normal_(self.flatten_nli_linear1.weight, nonlinearity='relu')
        init.xavier_uniform_(self.linear2.weight)

    def pooling(self, input, mode):
        if mode == 'CLS':
            pooled = input[:, 0, :]
        elif mode == 'MEAN':
            pooled = input.mean(dim=1)
        elif mode == 'SUM':
            pooled = input.sum(dim=1)
        elif mode == 'FLATTEN':
            pooled = input.view(input.shape[0], -1)
        else:
            raise NotImplementedError('Pooling type is not supported.')
        
        return pooled

    def forward(self, encoded_1, encoded_2) -> Tensor:
        pooled_1 = self.pooling(encoded_1, self.pooling_type)
        pooled_2 = self.pooling(encoded_2, self.pooling_type)
        if self.interaction == 'NLI':
            # NLI interaction style
            pooled = torch.concat([pooled_1, pooled_2, 
                                   pooled_1 * pooled_2, 
                                   pooled_1 - pooled_2], dim=-1)
            if self.pooling_type == 'FLATTEN':
                pooled = self.flatten_nli_linear1(pooled)
            else:
                pooled = self.nli_linear1(pooled)
        else:
            pooled = torch.concat([pooled_1, pooled_2], dim=-1)
            if self.pooling_type == 'FLATTEN':
                pooled = self.flatten_linear1(pooled)
            else:
                pooled = self.linear1(pooled)
        # pooled = self.relu(pooled)
        pooled = self.gelu(pooled)
        classified = self.linear2(pooled)
        classified = self.logsoftmax(classified)

        return classified


class ConverterLRASingle(nn.Module):
    def __init__(self, args) -> None:
        super(ConverterLRASingle, self).__init__()
        self.converter = converter.Converter(args)
        self.classifier = SingleClassifier(args.pooling_type, 
                                           args.max_seq_len, 
                                           args.encoder_dim, 
                                           args.mlp_dim, 
                                           args.num_class
                                           )

    def forward(self, input) -> Tensor:
        encoded = self.converter(input)
        classified = self.classifier(encoded)

        return classified

class ConverterLRADual(nn.Module):
    def __init__(self, args) -> None:
        super(ConverterLRADual, self).__init__()
        self.converter = converter.Converter(args)
        self.classifier = DualClassifier(args.pooling_type, 
                                         args.max_seq_len, 
                                         args.encoder_dim, 
                                         args.mlp_dim, 
                                         args.num_class
                                         )

    def forward(self, input1, input2) -> Tensor:
        encoded1 = self.converter(input1)
        encoded2 = self.converter(input2)
        classified = self.classifier(encoded1, encoded2)

        return classified