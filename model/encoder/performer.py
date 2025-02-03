import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from contextlib import contextmanager
from distutils.version import LooseVersion
from einops import repeat
from functools import partial
from torch import Tensor
from .. import embedding


TORCH_GE_1_8_0 = LooseVersion(torch.__version__) >= LooseVersion('1.8.0')

# helper functions

def exists(val):
    return val is not None

def empty(tensor):
    return tensor.numel() == 0

def default(val, d):
    return val if exists(val) else d

@contextmanager
def null_context():
    yield

def cast_tuple(val):
    return (val,) if not isinstance(val, tuple) else val

def get_module_device(module):
    return next(module.parameters()).device

def find_modules(nn_module, type):
    return [module for module in nn_module.modules() if isinstance(module, type)]

class Always(nn.Module):
    def __init__(self, val):
        super().__init__()
        self.val = val

    def forward(self, *args, **kwargs):
        return self.val

# token shifting helper and classes

def shift(t, amount, mask = None):
    if amount == 0:
        return t

    if exists(mask):
        t = t.masked_fill(~mask[..., None], 0.)

    return F.pad(t, (0, 0, amount, -amount), value = 0.)

class PreShiftTokens(nn.Module):
    def __init__(self, shifts, fn):
        super().__init__()
        self.fn = fn
        self.shifts = tuple(shifts)

    def forward(self, x, **kwargs):
        mask = kwargs.get('mask', None)
        shifts = self.shifts
        segments = len(shifts)
        feats_per_shift = x.shape[-1] // segments
        splitted = x.split(feats_per_shift, dim = -1)
        segments_to_shift, rest = splitted[:segments], splitted[segments:]
        segments_to_shift = list(map(lambda args: shift(*args, mask = mask), zip(segments_to_shift, shifts)))
        x = torch.cat((*segments_to_shift, *rest), dim = -1)
        return self.fn(x, **kwargs)
    
# kernel functions

def softmax_kernel(data, *, projection_matrix, is_query, normalize_data=True, eps=1e-4, device = None):
    b, h, *_ = data.shape

    data_normalizer = (data.shape[-1] ** -0.25) if normalize_data else 1.

    ratio = (projection_matrix.shape[0] ** -0.5)

    projection = repeat(projection_matrix, 'j d -> b h j d', b = b, h = h)
    projection = projection.type_as(data)

    data_dash = torch.einsum('...id,...jd->...ij', (data_normalizer * data), projection)

    diag_data = data ** 2
    diag_data = torch.sum(diag_data, dim=-1)
    diag_data = (diag_data / 2.0) * (data_normalizer ** 2)
    diag_data = diag_data.unsqueeze(dim=-1)

    if is_query:
        data_dash = ratio * (
            torch.exp(data_dash - diag_data -
                    torch.amax(data_dash, dim=-1, keepdim=True).detach()) + eps)
    else:
        data_dash = ratio * (
            torch.exp(data_dash - diag_data - torch.amax(data_dash, dim=(-1, -2), keepdim=True).detach()) + eps)

    return data_dash.type_as(data)

def generalized_kernel(data, *, projection_matrix, kernel_fn = nn.ReLU(), kernel_epsilon = 0.001, normalize_data = True, device = None):
    b, h, *_ = data.shape

    data_normalizer = (data.shape[-1] ** -0.25) if normalize_data else 1.

    if projection_matrix is None:
        return kernel_fn(data_normalizer * data) + kernel_epsilon

    projection = repeat(projection_matrix, 'j d -> b h j d', b = b, h = h)
    projection = projection.type_as(data)

    data_dash = torch.einsum('...id,...jd->...ij', (data_normalizer * data), projection)

    data_prime = kernel_fn(data_dash) + kernel_epsilon
    return data_prime.type_as(data)

def orthogonal_matrix_chunk(cols, device = None):
    unstructured_block = torch.randn((cols, cols), device = device)
    if TORCH_GE_1_8_0:
        q, r = torch.linalg.qr(unstructured_block.cpu(), mode = 'reduced')
    else:
        q, r = torch.qr(unstructured_block.cpu(), some = True)
    q, r = map(lambda t: t.to(device), (q, r))
    return q.t()

def gaussian_orthogonal_random_matrix(nb_rows, nb_columns, scaling = 0, device = None):
    nb_full_blocks = int(nb_rows / nb_columns)

    block_list = []

    for _ in range(nb_full_blocks):
        q = orthogonal_matrix_chunk(nb_columns, device = device)
        block_list.append(q)

    remaining_rows = nb_rows - nb_full_blocks * nb_columns
    if remaining_rows > 0:
        q = orthogonal_matrix_chunk(nb_columns, device = device)
        block_list.append(q[:remaining_rows])

    final_matrix = torch.cat(block_list)

    if scaling == 0:
        multiplier = torch.randn((nb_rows, nb_columns), device = device).norm(dim = 1)
    elif scaling == 1:
        multiplier = math.sqrt((float(nb_columns))) * torch.ones((nb_rows,), device = device)
    else:
        raise ValueError(f'Invalid scaling {scaling}')

    return torch.diag(multiplier) @ final_matrix

def linear_attention(q, k, v):
    k_cumsum = k.sum(dim = -2)
    D_inv = 1. / torch.einsum('...nd,...d->...n', q, k_cumsum.type_as(q))
    context = torch.einsum('...nd,...ne->...de', k, v)
    out = torch.einsum('...de,...nd,...n->...ne', context, q, D_inv)
    return out

class PerformerMultiHeadAttention(nn.Module):
    def __init__(self, 
                 feat_dim: int,
                 num_heads: int, 
                 ortho_scaling: float = 0.0,
                 generalized_attention: bool = False, 
                 kernel_fn: str = 'relu', 
                 no_projection: bool = False,
                 value_drop_prob: float = 0.1,
                 ):
        super(PerformerMultiHeadAttention, self).__init__()
        dim_heads = feat_dim // num_heads
        self.nb_features = int(dim_heads * math.log(dim_heads))

        self.create_projection = partial(gaussian_orthogonal_random_matrix, nb_rows = self.nb_features, nb_columns = dim_heads, scaling = ortho_scaling)
        projection_matrix = self.create_projection()
        self.register_buffer('projection_matrix', projection_matrix)

        self.generalized_attention = generalized_attention

        if kernel_fn == 'relu':
            self.kernel_fn = nn.ReLU()
        elif kernel_fn == 'softplus':
            self.kernel_fn = nn.Softplus()

        # if this is turned on, no projection will be used
        # queries and keys will be softmax-ed as in the original efficient attention paper
        self.no_projection = no_projection

        self.value_dropout = nn.Dropout(p=value_drop_prob)
        self.query_linear = nn.Linear(feat_dim, feat_dim, bias=False)
        self.key_linear = nn.Linear(feat_dim, feat_dim, bias=False)
        self.value_linear = nn.Linear(feat_dim, feat_dim, bias=False)
        self.output_linear = nn.Linear(feat_dim, feat_dim, bias=False)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.xavier_uniform_(self.query_linear.weight, gain=1.0)
        init.xavier_uniform_(self.key_linear.weight, gain=1.0)
        init.xavier_uniform_(self.value_linear.weight, gain=1.0)
        init.xavier_uniform_(self.output_linear.weight, gain=1.0)

    @torch.no_grad()
    def redraw_projection_matrix(self, device):
        projections = self.create_projection(device = device)
        self.projection_matrix.copy_(projections)
        del projections

    def forward(self, input : Tensor) -> Tensor:
        q, k, v = self.query_linear(input), self.key_linear(input), self.value_linear(input)
        v = self.value_dropout(v)

        if self.no_projection:
            q = q.softmax(dim = -1)
            k = k.softmax(dim = -2)
        elif self.generalized_attention:
            create_kernel = partial(generalized_kernel, kernel_fn = self.kernel_fn, projection_matrix = self.projection_matrix, device = input.device)
            q, k = map(create_kernel, (q, k))
        else:
            create_kernel = partial(softmax_kernel, projection_matrix = self.projection_matrix, device = input.device)
            q = create_kernel(q, is_query = True)
            k = create_kernel(k, is_query = False)

        attn_fn = linear_attention
        out = attn_fn(q, k, v)
        return out
    

class PreLayerNorm(nn.Module):
    def __init__(self, dim, func):
        super().__init__()
        self.layer_norm = nn.LayerNorm(dim)
        self.func = func

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        return self.layer_norm(self.func(x, **kwargs)) + x


class FeedForwardNetwork(nn.Module):
    def __init__(self, feat_dim: int, hidden_dim: int, ffn_drop_prob: float = 0.1) -> None:
        super(FeedForwardNetwork, self).__init__()
        self.ffn = nn.Sequential(
            nn.Linear(in_features=feat_dim, out_features=hidden_dim, bias=True),
            nn.GELU(),
            nn.Dropout(p=ffn_drop_prob),
            nn.Linear(in_features=hidden_dim, out_features=feat_dim, bias=True)
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_normal_(self.ffn[0].weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
        init.kaiming_normal_(self.ffn[3].weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, input: Tensor) -> Tensor:
        ffn_output = self.ffn(input)

        return ffn_output


class PerformerEncoder(nn.Module):
    def __init__(self, args):
        super(PerformerEncoder, self).__init__()
        self.embedding = embedding.Embedding(args.pe_type, 
                                             args.pooling_type, 
                                             args.vocab_size, 
                                             args.max_seq_len, 
                                             args.embed_dim, 
                                             args.pe_drop_prob, 
                                             args.embed_drop_prob)
        self.performer_encoder_block = nn.ModuleList([])
        for _ in range(args.num_block):
            self.performer_encoder_block.append(nn.ModuleList([
                PreLayerNorm(args.embed_dim, PerformerMultiHeadAttention(args.embed_dim, args.num_head, args.xformer.performer.ortho_scaling, args.xformer.performer.kernel_func, args.xformer.performer.no_projection, args.value_drop_prob)),
                PreLayerNorm(args.embed_dim, FeedForwardNetwork(args.embed_dim, args.hidden_dim, args.ffn_drop_prob))
            ]))

    def forward(self, input: Tensor) -> Tensor:
        input = self.embedding(input)

        for mhpa, ffn in self.performer_encoder_block:
            input = mhpa(input)
            input = ffn(input)
        
        return input