import torch
import torch.nn as nn
import torch.nn.init as init
from . import embedding, linformer
from torch import Tensor


class SingleClassifier(nn.Module):
    def __init__(self, pooling_type, max_seq_len, encoder_dim, mlp_dim, num_class) -> None:
        super(SingleClassifier, self).__init__()
        self.pooling_type = pooling_type
        self.max_seq_len = max_seq_len
        self.encoder_dim = encoder_dim
        self.mlp_dim = mlp_dim
        self.num_class = num_class
        if pooling_type == 'FLATTEN':
            self.linear1 = nn.Linear(in_features=max_seq_len * encoder_dim, out_features=mlp_dim, bias=True)
        else:
            self.linear1 = nn.Linear(in_features=encoder_dim, out_features=mlp_dim, bias=True)
        self.linear2 = nn.Linear(in_features=mlp_dim, out_features=num_class, bias=True)
        self.leaky_relu = nn.LeakyReLU()

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_normal_(self.linear1.weight, nonlinearity='leaky_relu')
        init.xavier_normal_(self.linear2.weight, gain=1.0)
        init.zeros_(self.linear1.bias)
        init.zeros_(self.linear2.bias)
    
    def pooling(self, input: Tensor, mode: str) -> Tensor:
        if mode == 'CLS':
            pooled = input[:, 0, :]
        elif mode == 'MEAN':
            pooled = input.mean(dim=1)
        elif mode == 'SUM':
            pooled = input.sum(dim=1)
        elif mode == 'FLATTEN':
            pooled = input.contiguous().view(input.shape[0], -1)
        else:
            raise NotImplementedError('Pooling type is not supported.')
        
        return pooled

    def forward(self, encoded: Tensor) -> Tensor:
        pooled = self.pooling(encoded, self.pooling_type)
        pooled1 = self.linear1(pooled)
        pooled1 = self.leaky_relu(pooled1)
        pooled2 = self.linear2(pooled1)

        return pooled2


class DualClassifier(nn.Module):
    def __init__(self, pooling_type, max_seq_len, encoder_dim, mlp_dim, num_class, interaction) -> None:
        super(DualClassifier, self).__init__()
        self.pooling_type = pooling_type
        self.interaction = interaction
        self.max_seq_len = max_seq_len
        self.encoder_dim = encoder_dim
        self.mlp_dim = mlp_dim
        self.num_class = num_class
        if interaction == 'CAT':
            if pooling_type == 'FLATTEN':
                self.linear1 = nn.Linear(in_features=max_seq_len * encoder_dim * 2, out_features=mlp_dim, bias=True)
            else:
                self.linear1 = nn.Linear(in_features=encoder_dim * 2, out_features=mlp_dim, bias=True)
        else:
            if pooling_type == 'FLATTEN':
                self.linear1 = nn.Linear(in_features=max_seq_len * encoder_dim * 4, out_features=mlp_dim, bias=True)
            else:
                self.linear1 = nn.Linear(in_features=encoder_dim * 4, out_features=mlp_dim, bias=True)
        self.linear2 = nn.Linear(in_features=mlp_dim, out_features=num_class, bias=True)
        self.leaky_relu = nn.LeakyReLU()

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_normal_(self.linear1.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
        init.xavier_normal_(self.linear2.weight, gain=1.0)
        init.zeros_(self.linear1.bias)
        init.zeros_(self.linear2.bias)

    def pooling(self, input: Tensor, mode: str) -> Tensor:
        if mode == 'CLS':
            pooled = input[:, 0, :]
        elif mode == 'MEAN':
            pooled = input.mean(dim=1)
        elif mode == 'SUM':
            pooled = input.sum(dim=1)
        elif mode == 'FLATTEN':
            pooled = input.contiguous().view(input.shape[0], -1)
        else:
            raise NotImplementedError('Pooling type is not supported.')
        
        return pooled

    def forward(self, encoded_1: Tensor, encoded_2: Tensor) -> Tensor:
        pooled_1 = self.pooling(encoded_1, self.pooling_type)
        pooled_2 = self.pooling(encoded_2, self.pooling_type)
        if self.interaction == 'NLI':
            # NLI interaction style
            pooled = torch.cat([pooled_1, 
                                pooled_2, 
                                pooled_1 * pooled_2, 
                                pooled_1 - pooled_2], dim=-1)
            pooled_layer1 = self.linear1(pooled)
        else:
            pooled = torch.cat([pooled_1, pooled_2], dim=-1)
            pooled_layer1 = self.linear1(pooled)
        pooled_layer1 = self.leaky_relu(pooled_layer1)
        pooled_layer2 = self.linear2(pooled_layer1)

        return pooled_layer2


class LRASingle(nn.Module):
    def __init__(self, args) -> None:
        super(LRASingle, self).__init__()
        self.embedding = embedding.Embedding(args.pe_type, 
                                             args.pooling_type, 
                                             args.vocab_size, 
                                             args.max_seq_len, 
                                             args.embed_size, 
                                             args.embed_drop_prob)
        self.linformer = linformer.Linformer(args)
        self.classifier = SingleClassifier(args.pooling_type, 
                                           args.max_seq_len, 
                                           args.encoder_dim, 
                                           args.mlp_dim, 
                                           args.num_class
                                           )

    def forward(self, input: Tensor) -> Tensor:
        embeded = self.embedding(input)
        encoded = self.linformer(embeded)
        classified = self.classifier(encoded)

        return classified


class LRADual(nn.Module):
    def __init__(self, args) -> None:
        super(LRADual, self).__init__()
        self.embedding = embedding.Embedding(args.pe_type, 
                                             args.pooling_type, 
                                             args.vocab_size, 
                                             args.max_seq_len, 
                                             args.embed_size, 
                                             args.embed_drop_prob)
        self.linformer = linformer.Linformer(args)
        self.classifier = DualClassifier(args.pooling_type, 
                                         args.max_seq_len, 
                                         args.encoder_dim, 
                                         args.mlp_dim, 
                                         args.num_class,
                                         args.interaction
                                         )

    def forward(self, input1: Tensor, input2: Tensor) -> Tensor:
        embeded1 = self.embedding(input1)
        embeded2 = self.embedding(input2)
        encoded1 = self.linformer(embeded1)
        encoded2 = self.linformer(embeded2)
        classified = self.classifier(encoded1, encoded2)

        return classified