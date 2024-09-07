import pandas as pd
import torch
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torchtext import data


def df_tsv(path, header=True, columns_dict={}):
    a=[]
    with open(path, 'r') as f:
        for i in f:
            a+=[i[:-1].split('\t')]
    if header:
        res=pd.DataFrame(a[1:], columns=a[0])
    else:
        res=pd.DataFrame(a)
  
    if len(columns_dict)>0:
        res.columns=list(columns_dict.keys())
        res=res.rename(columns=columns_dict)
        res=res.copy().drop('', axis=1, errors='ignore')

    return res


def prep_inputs_masks(df, category_dict, tokenizer, task, MAX_LEN=128):
    sentences = df.sent.values

    lables = df.label.apply(lambda x: category_dict[x]).values

    if task == 'sentiment':
        sentences = ['[CLS] ' + sentences[idx] + ' [SEP]' for idx in range(len(sentences))]
    elif task == 'similarity':
        sentences1 = df.sent1.values
        sentences = ['[CLS] ' + sentences[idx] + ' [SEP] ' + sentences1[idx] + ' [SEP]' for idx in range(len(sentences))]
    else:
        raise ValueError(f'ERROR: The {task} task is a wrong task.')
    
    tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]

    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]

    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    input_ids = tokenizer.pad(
        input_ids,
        padding='max_length',  
        max_length=MAX_LEN,    
        truncation=True,       
        return_tensors="pt"    
    )['input_ids']  

    input_ids = input_ids.long()  

    return input_ids, lables


def gen_dataloader(train_inputs, train_labels, batch_size = 32):
    train_data = TensorDataset(train_inputs, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size,pin_memory=True,num_workers=4)

    return train_dataloader


def datagen_BERT(path, header, columns_dict, category_dict, tokenizer, task, batch_size, max_len=128):

    df_train=df_tsv(path, header=header, columns_dict=columns_dict)

    train_inputs, train_labels=prep_inputs_masks(df_train, category_dict, tokenizer, task=task, MAX_LEN=max_len)
    train_inputs = torch.tensor(train_inputs)
    train_labels = torch.tensor(train_labels)
    train_dataloader=gen_dataloader(train_inputs, train_labels, batch_size = batch_size)

    return train_dataloader, train_inputs, df_train


def load_BERT_data(paths, tokenizer, task_type, device, BATCH_SIZE=128, seed=1234):
    init_token_idx = tokenizer.cls_token_id
    eos_token_idx = tokenizer.sep_token_id
    pad_token_idx = tokenizer.pad_token_id
    unk_token_idx = tokenizer.unk_token_id

    max_input_length = tokenizer.max_model_input_sizes['bert-base-uncased']

    def tokenize_and_cut(sentence):
        tokens = tokenizer.tokenize(sentence) 
        tokens = tokens[:max_input_length-2]

        return tokens
  
    TEXT = data.Field(batch_first = True, 
                      use_vocab = False, 
                      tokenize = tokenize_and_cut, 
                      preprocessing = tokenizer.convert_tokens_to_ids, 
                      init_token = init_token_idx, 
                      eos_token = eos_token_idx, 
                      pad_token = pad_token_idx, 
                      unk_token = unk_token_idx)

    LABEL = data.LabelField(dtype = torch.float)
  
    for key in paths.keys():
        for i,label_value in enumerate(paths[key]['schema']):
            if label_value[1]=='LABEL':
                paths[key]['schema'][i]=(label_value[0],LABEL)
            elif label_value[1]=='TEXT':
                paths[key]['schema'][i]=(label_value[0],TEXT)

    trn_path=paths['train']['path']
    trn_schema=paths['train']['schema']
    try:
        val_path=paths['validation']['path']
        val_schema=paths['validation']['schema']
        val_split=True
    except:
        val_split=False
    tst_path=paths['test']['path']
    tst_schema=paths['test']['schema']

    if val_split:
        trn_data=data.TabularDataset(trn_path,'tsv', fields=trn_schema)
        val_data=data.TabularDataset(val_path,'tsv', fields=val_schema)
        tst_data=data.TabularDataset(tst_path,'tsv', fields=tst_schema)
    else:
        trn_data, val_data = data.TabularDataset(trn_path,'tsv', fields=trn_schema)
        tst_data = data.TabularDataset(tst_path,'tsv', fields=tst_schema)

    assert(device==torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    LABEL.build_vocab(trn_data)
    trn_iter, val_iter, tst_iter = data.Iterator.splits(
        (trn_data, val_data, tst_data), 
        sort=False, 
        batch_size = BATCH_SIZE, 
        device = device)


    return trn_iter, val_iter, tst_iter, LABEL, TEXT