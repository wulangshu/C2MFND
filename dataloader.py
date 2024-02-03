import torch
import random
import pandas as pd
import numpy as np
import pickle
from config import CDRDConfig
from transformers import BertTokenizer,RobertaTokenizer,AutoTokenizer
from torch.utils.data import TensorDataset, DataLoader
from copy import deepcopy



def read_pkl(path):
    with open(path, "rb") as f:
        t = pickle.load(f)
    return t

def df_filter(df_data):
    df_data = df_data[df_data['category'] != '无法确定']
    return df_data

def word2input(texts, bert, max_len,language):
    if language=='cn':
        tokenizer = BertTokenizer(vocab_file=bert+'/vocab.txt')
        token_ids = []
        for i, text in enumerate(texts):
            token_ids.append(
                tokenizer.encode(text, max_length=max_len, add_special_tokens=True, padding='max_length',
                                truncation=True))
        token_ids = torch.tensor(token_ids)
        masks = torch.zeros(token_ids.shape)
        mask_token_id = tokenizer.pad_token_id
        for i, tokens in enumerate(token_ids):
            masks[i] = (tokens != mask_token_id)
        return token_ids, masks, torch.tensor([[float('nan')]]*token_ids.shape[0])
   
    elif language=='en':
        tokenizer = RobertaTokenizer.from_pretrained(bert)
        token_ids = []
        for i, text in enumerate(texts):
            token_ids.append(
                tokenizer.encode(text, max_length=max_len, add_special_tokens=True, padding='max_length',
                                truncation=True))
        token_ids = torch.tensor(token_ids)
        masks = torch.zeros(token_ids.shape)
        mask_token_id = tokenizer.pad_token_id
        for i, tokens in enumerate(token_ids):
            masks[i] = (tokens != mask_token_id)
        return token_ids, masks, torch.tensor([[float('nan')]]*token_ids.shape[0])
   
    else: 
        tokenizer = AutoTokenizer.from_pretrained(bert)
        token_ids = []
        for i, text in enumerate(texts):
            token_ids.append(
                tokenizer.encode(text, max_length=max_len, add_special_tokens=True, padding='max_length',
                                truncation=True))
        token_ids = torch.tensor(token_ids)
        masks = torch.zeros(token_ids.shape)
        mask_token_id = tokenizer.pad_token_id
        for i, tokens in enumerate(token_ids):
            masks[i] = (tokens != mask_token_id)

        global_attention_mask = torch.zeros(
            token_ids.shape, dtype=torch.long, device=token_ids.device
        )  # initialize to global attention to be deactivated for all tokens
        global_attention_mask[
            :,
            [
                1,
                4,
                21,
            ],
        ] = 1  # Set global attention to random tokens for the sake of this example
        # Usually, set global attention based on the task. For example,
        # classification: the <s> token
        # QA: question tokens
        # LM: potentially on the beginning of sentences and paragraphs
        return token_ids, masks, global_attention_mask
    

       
        
class bert_data():
    def __init__(self, max_len, batch_size, pool_size, bert, category_dict,category_count,language):
        self.max_len = max_len
        self.batch_size = batch_size
        self.pool_size= pool_size
        self.bert = bert
        self.category_dict = category_dict
        self.category_count = category_count
        self.language=language
    
    def load_data(self, path, shuffle):
        self.data = df_filter(read_pkl(path))
        
        count_type=deepcopy(self.category_count)
        count_type_pos=deepcopy(self.category_count)
        count_type_neg=deepcopy(self.category_count)
        proportion={}
        for i in range(len(self.data)):
            count_type[self.data['category'][i]]+=1
            if self.data['label'][i]==1:
                count_type_pos[self.data['category'][i]]+=1
            elif self.data['label'][i]==0:
                count_type_neg[self.data['category'][i]]+=1
        for i in count_type.keys():
            proportion[self.category_dict[i]]=count_type_pos[i]/count_type[i]
        

        content = self.data['content'].to_numpy()
        
        label = torch.tensor(self.data['label'].astype(int).to_numpy())
        category = torch.tensor(self.data['category'].apply(lambda c: self.category_dict[c]).to_numpy())
        
       
        content_emotion = torch.tensor(np.vstack(self.data['content_emotion']).astype('float32'))
        comments_emotion = torch.tensor(np.vstack(self.data['comments_emotion']).astype('float32'))
        emotion_gap = torch.tensor(np.vstack(self.data['emotion_gap']).astype('float32'))
        style_feature = torch.tensor(np.vstack(self.data['style_feature']).astype('float32'))
        
        
        content_token_ids, content_masks,content_global_attention_mask = word2input(content, self.bert, self.max_len,self.language)

        dataset = TensorDataset(content_token_ids,
                                content_masks,
                                content_global_attention_mask,
                                content_emotion,
                                comments_emotion,
                                emotion_gap,
                                style_feature,
                                label,
                                category
                                )
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            pin_memory=True,#锁页内存
            shuffle=shuffle 
        )
        return dataloader,proportion
