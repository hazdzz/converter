config = {
    "listops":{
        "pe_type": "cope", # "nope", "spe", "ape", or "cope"
        "vocab_size": 15 + 1 + 1, # 15 tokens + 1 PAD + 1 CLS
        "embed_dim": 32,
        "max_seq_len": 1999 + 1,
        "digraphconv_type": "chsyconv1d",
        "enable_kpm": True,
        "enable_kploss": True,
        "kernel_type": "none", # 'none', 'dirichlet', 'fejer', 'jackson', 'lanczos', 'lorentz', 'vekic', or 'wang'
        "max_order": 2,
        "mu": 3,
        "xi": 4.0,
        "stigma": 0.5,
        "heta": 2,
        "dataset_name": "listops",
        "pooling_type": "CLS", # "CLS", "MEAN", "SUM", or "FLATTEN"
        "encoder_dim": 32,
        "mlp_dim": 32,
        "num_class": 10,
        "interaction": "None",
        "enable_cuda": True,
        "device_id": 0,
        "embed_drop_prob": 0.0,
        "eigenvalue_drop_prob": 0.0,
        "value_drop_prob": 0.1,
        "bffn_drop_prob": 0.1,
        "batch_size": 64,
        "lr": 0.001,
        "weight_decay": 0.001,
        "epochs": 40,
        "optimizer": "lion", # "adamw", "lion", "tiger", or "sophia"
        "patience": 2,
        "num_workers": 2
    },
    "image":{
        "pe_type": "cope", # "nope", "spe", "ape", or "cope"
        "vocab_size": 256, # 256 unique pixel values
        "embed_dim": 64,
        "max_seq_len": 1024,
        "digraphconv_type": "chsyconv1d",
        "enable_kpm": True,
        "enable_kploss": True,
        "kernel_type": "none", # 'none', 'dirichlet', 'fejer', 'jackson', 'lanczos', 'lorentz', 'vekic', or 'wang'
        "max_order": 2,
        "mu": 3,
        "xi": 4.0,
        "stigma": 0.5,
        "heta": 2,
        "dataset_name": "image",
        "pooling_type": "FLATTEN", # "CLS", "MEAN", "SUM", or "FLATTEN"
        "encoder_dim": 64,
        "mlp_dim": 64,
        "num_class": 10,
        "interaction": "None", 
        "enable_cuda": True,
        "device_id": 0, # single GPU
        "embed_drop_prob": 0.0,
        "eigenvalue_drop_prob": 0.0,
        "value_drop_prob": 0.0,
        "bffn_drop_prob": 0.1,
        "batch_size": 64,
        "lr": 0.001,
        "weight_decay": 0.001,
        "epochs": 20,
        "optimizer": "lion", # "adamw", "lion", "tiger", or "sophia"
        "patience": 2,
        "num_workers": 2
    },
    "pathfinder":{
        "pe_type": "cope", # "nope", "spe", "ape", or "cope"
        "vocab_size": 225, # 225 unique pixel values
        "embed_dim": 64,
        "max_seq_len": 1024,
        "digraphconv_type": "chsyconv1d",
        "enable_kpm": True,
        "enable_kploss": True,
        "kernel_type": "none", # 'none', 'dirichlet', 'fejer', 'jackson', 'lanczos', 'lorentz', 'vekic', or 'wang'
        "max_order": 2,
        "mu": 3,
        "xi": 4.0,
        "stigma": 0.5,
        "heta": 2,
        "dataset_name": "pathfinder",
        "pooling_type": "FLATTEN", # "CLS", "MEAN", "SUM", or "FLATTEN"
        "encoder_dim": 64,
        "mlp_dim": 64,
        "num_class": 2,
        "interaction": "None",
        "enable_cuda": True,
        "device_id": 0,
        "embed_drop_prob": 0.0,
        "eigenvalue_drop_prob": 0.0,
        "value_drop_prob": 0.1,
        "bffn_drop_prob": 0.1,
        "batch_size": 256,
        "lr": 0.001,
        "weight_decay": 0.001,
        "epochs": 50,
        "optimizer": "lion", # "adamw", "lion", "tiger", or "sophia"
        "patience": 2,
        "num_workers": 2
    },
    "text":{
        "pe_type": "cope", # "nope", "spe", "ape", or "cope"
        "vocab_size": 95 + 1 + 1, # 95 unique symbols + 1 PAD + 1 CLS
        "embed_dim": 64,
        "max_seq_len": 4096 + 1,
        "digraphconv_type": "chsyconv1d",
        "enable_kpm": True,
        "enable_kploss": True,
        "kernel_type": "none", # 'none', 'dirichlet', 'fejer', 'jackson', 'lanczos', 'lorentz', 'vekic', or 'wang'
        "max_order": 2,
        "mu": 3,
        "xi": 4.0,
        "stigma": 0.5,
        "heta": 2,
        "dataset_name": "text",
        "pooling_type": "CLS", # "CLS", "MEAN", "SUM", or "FLATTEN"
        "encoder_dim": 64,
        "mlp_dim": 64,
        "num_class": 2,
        "interaction": "None",
        "enable_cuda": True,
        "device_id": 0,
        "embed_drop_prob": 0.0,
        "eigenvalue_drop_prob": 0.0,
        "value_drop_prob": 0.0,
        "bffn_drop_prob": 0.1,
        "batch_size": 16,
        "lr": 0.001,
        "weight_decay": 0.001,
        "epochs": 30,
        "optimizer": "lion", # "adamw", "lion", "tiger", or "sophia"
        "patience": 2,
        "num_workers": 2
    },
    "retrieval":{
        "pe_type": "cope", # "nope", "spe", "ape", or "cope"
        "vocab_size": 96 + 1 + 1, # 96 unique symbols + 1 PAD + 1 CLS
        "embed_dim": 64,
        "max_seq_len": 4000 + 1,
        "digraphconv_type": "chsyconv1d",
        "enable_kpm": True,
        "enable_kploss": True,
        "kernel_type": "none", # 'none', 'dirichlet', 'fejer', 'jackson', 'lanczos', 'lorentz', 'vekic', or 'wang'
        "max_order": 3,
        "mu": 3,
        "xi": 4.0,
        "stigma": 0.5,
        "heta": 2,
        "dataset_name": "retrieval",
        "pooling_type": "CLS", # "CLS", "MEAN", "SUM", or "FLATTEN"
        "encoder_dim": 64,
        "mlp_dim": 64,
        "num_class": 2,
        "interaction": "NLI", # "NLI" or "CAT"
        "enable_cuda": True,
        "device_id": 0,
        "embed_drop_prob": 0.0,
        "eigenvalue_drop_prob": 0.0,
        "value_drop_prob": 0.1,
        "bffn_drop_prob": 0.1,
        "batch_size": 256,
        "lr": 0.001,
        "weight_decay": 0.001,
        "epochs": 20,
        "optimizer": "lion", # "adamw", "lion", "tiger", or "sophia"
        "patience": 2,
        "num_workers": 2
    }
}