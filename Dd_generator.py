import torch
import torch.nn as nn
import random
import numpy as np
import os
from dataloader import read_pkl,df_filter,word2input
from torch.utils.data import TensorDataset
from config import CDRDConfig
from transformers import BertModel,RobertaModel,LongformerModel
from layer import Cnn_extractor,MLP,CrossAttention
from utils import Buffer,getfeature_domain
import pickle


def data2gpu(batch, use_cuda):
    if use_cuda:
        batch_data = {
            'content': batch[0].cuda(),
            'content_masks': batch[1].cuda(),
            'content_global_attention_mask': batch[2].cuda(),
            'content_emotion': batch[3].cuda(),
            'comments_emotion': batch[4].cuda(),
            'emotion_gap': batch[5].cuda(),
            'style_feature': batch[6].cuda()
            }
    else:
        batch_data = {
            'content': batch[0],
            'content_masks': batch[1],
            'content_global_attention_mask': batch[2],
            'content_emotion': batch[3],
            'comments_emotion': batch[4],
            'emotion_gap': batch[5],
            'style_feature': batch[6]
            }
    return batch_data

class Generator(nn.Module):
    def __init__(self,feature_kernel,emb_dim,middel_dim,W_dim,head_num,ffn_dim,dropout,domain_num,bert,semantic_extractor,semantic_num,emotion_extractor,emotion_num,style_extractor,style_num,weight,crossattention,buffer,language):
        super(Generator,self).__init__()

        self.semantic_num=semantic_num
        self.emotion_num=emotion_num
        self.style_num=style_num
        self.mid_dim=middel_dim*1
        if language=='cn':
            self.bert=BertModel.from_pretrained(bert).requires_grad_(False)
        elif language=='en':
            self.bert=RobertaModel.from_pretrained(bert).requires_grad_(False)
        else:
            self.bert = LongformerModel.from_pretrained(bert).requires_grad_(False)
        
        self.semantic_extractor=nn.ModuleList([Cnn_extractor(feature_kernel,emb_dim) for i in range(semantic_num)])
        if language=='cn':
            self.emotion_extractor=nn.ModuleList([MLP(47*5,[int(0.8*middel_dim),middel_dim],dropout) for i in range(emotion_num)])
            self.style_extractor = nn.ModuleList([MLP(48,[int(0.8*middel_dim),middel_dim],dropout) for i in range(style_num)])
        else:
            self.emotion_extractor=nn.ModuleList([MLP(38*5,[int(0.8*middel_dim),middel_dim],dropout) for i in range(emotion_num)])
            self.style_extractor = nn.ModuleList([MLP(32,[int(0.8*middel_dim),middel_dim],dropout) for i in range(style_num)])
        
        self.crossattention=CrossAttention(self.mid_dim,int(W_dim*self.mid_dim),head_num,ffn_dim)

        self.weight=torch.load(weight).requires_grad_(False) 
        
        with open(buffer, 'rb') as file:
            self.buffer = pickle.load(file)

        self.semantic_extractor.load_state_dict(torch.load(semantic_extractor))
        self.emotion_extractor.load_state_dict(torch.load(emotion_extractor))
        self.style_extractor.load_state_dict(torch.load(style_extractor))
        self.crossattention.load_state_dict(torch.load(crossattention))
        self.semantic_extractor.requires_grad_(False)
        self.emotion_extractor.requires_grad_(False)
        self.style_extractor.requires_grad_(False)
        self.crossattention.requires_grad_(False)
        for i in range(domain_num):
            self.buffer.pool[i].requires_grad_(False)

    def forward(self,tag,**kwargs):
        inputs = kwargs['content']
        masks = kwargs['content_masks']
        global_attention_mask = kwargs['content_global_attention_mask']
        
        
        content_emotion = kwargs['content_emotion']
        comments_emotion = kwargs['comments_emotion']
        emotion_gap = kwargs['emotion_gap']
        style_feature = kwargs['style_feature']

        emotion_feature = torch.cat([content_emotion, comments_emotion, emotion_gap], dim=1)

        if global_attention_mask.isnan()._is_all_true():
            init_feature = self.bert (inputs, attention_mask = masks)[0]
        else:
            init_feature = self.bert (inputs, attention_mask = masks, global_attention_mask=global_attention_mask)[0]

        T_feature=[]
        T_feature.extend([self.semantic_extractor[i](init_feature).unsqueeze(1) for i in range(self.semantic_num)])
        T_feature.extend([self.emotion_extractor[i](emotion_feature).unsqueeze(1) for i in range(self.emotion_num)])
        T_feature.extend([self.style_extractor[i](style_feature).unsqueeze(1) for i in range(self.style_num)])
        T_feature=torch.cat(T_feature,dim=1)
        T_feature=torch.matmul(torch.softmax(self.weight,dim=-1),T_feature).squeeze(1)
        
        domain_old=getfeature_domain(self.buffer,tag)
        domain_new=self.crossattention(domain_old.detach(),T_feature.detach())
        
        return domain_new
    

class new_loader():
    def __init__(self,max_len,bert,category_dict,path,language):
        self.max_len = max_len
        self.bert = bert
        self.category_dict = category_dict
        self.category_dict_rvs={self.category_dict[i]:i for i in self.category_dict.keys()}
        

        data=df_filter(read_pkl(path))
        data_new={}
        for i in self.category_dict_rvs.keys():
            data_new[i]={}
            a,b=list(data['category']==self.category_dict_rvs[i]),list(data['label']==0)
            data_new[i][0]=data.loc[[x & y for x, y in zip(a, b)]]
            a,b=list(data['category']==self.category_dict_rvs[i]),list(data['label']==1)
            data_new[i][1]=data.loc[[x & y for x, y in zip(a, b)]]
        content,content_token_ids, content_masks,content_global_attention_mask,content_emotion,comments_emotion,emotion_gap,style_feature ={},{},{},{},{},{},{},{}
        self.dataset={}
        for i in self.category_dict_rvs.keys():
            content[i]={}
            content[i][0]=data_new[i][0]['content'].to_numpy()
            content[i][1]=data_new[i][1]['content'].to_numpy()
            
            content_token_ids[i], content_masks[i],content_global_attention_mask[i],content_emotion[i],comments_emotion[i],emotion_gap[i],style_feature[i]={},{},{},{},{},{},{}
            content_token_ids[i][0], content_masks[i][0] ,content_global_attention_mask[i][0]= word2input(content[i][0], self.bert, self.max_len,language)
            content_token_ids[i][1], content_masks[i][1] ,content_global_attention_mask[i][1]= word2input(content[i][1], self.bert, self.max_len,language)
           

            content_emotion[i][0] = torch.tensor(np.vstack(data_new[i][0]['content_emotion']).astype('float32'))
            content_emotion[i][1] = torch.tensor(np.vstack(data_new[i][1]['content_emotion']).astype('float32'))
            comments_emotion[i][0] = torch.tensor(np.vstack(data_new[i][0]['comments_emotion']).astype('float32'))
            comments_emotion[i][1] = torch.tensor(np.vstack(data_new[i][1]['comments_emotion']).astype('float32'))
            emotion_gap[i][0] = torch.tensor(np.vstack(data_new[i][0]['emotion_gap']).astype('float32'))
            emotion_gap[i][1] = torch.tensor(np.vstack(data_new[i][1]['emotion_gap']).astype('float32'))
            style_feature[i][0] = torch.tensor(np.vstack(data_new[i][0]['style_feature']).astype('float32'))
            style_feature[i][1] = torch.tensor(np.vstack(data_new[i][1]['style_feature']).astype('float32'))
           

            self.dataset[i]={}
            self.dataset[i][0]=TensorDataset(  content_token_ids[i][0],
                                            content_masks[i][0],
                                            content_global_attention_mask[i][0],
                                            content_emotion[i][0],
                                            comments_emotion[i][0],
                                            emotion_gap[i][0],
                                            style_feature[i][0]
                                            )
            self.dataset[i][1]=TensorDataset(  content_token_ids[i][1],
                                            content_masks[i][1],
                                            content_global_attention_mask[i][1],
                                            content_emotion[i][1],
                                            comments_emotion[i][1],
                                            emotion_gap[i][1],
                                            style_feature[i][1]
                                            )
        
            
    def produce(self,num_pos,num_neg,tag):
        pick_pos=torch.randint(0,len(self.dataset[tag][1]),(num_pos,)).tolist()
        pick_neg=torch.randint(0,len(self.dataset[tag][0]),(num_neg,)).tolist()
        content,content_mask,content_global_attention_mask,content_emotion,comments_emotion,emotion_gap,style_feature=[],[],[],[],[],[],[]
        for i in pick_pos:
            content.append(self.dataset[tag][1][i][0].unsqueeze(dim=0))
            content_mask.append(self.dataset[tag][1][i][1].unsqueeze(dim=0))
            content_global_attention_mask.append(self.dataset[tag][1][i][2].unsqueeze(dim=0))
            content_emotion.append(self.dataset[tag][1][i][3].unsqueeze(dim=0))
            comments_emotion.append(self.dataset[tag][1][i][4].unsqueeze(dim=0))
            emotion_gap.append(self.dataset[tag][1][i][5].unsqueeze(dim=0))
            style_feature.append(self.dataset[tag][1][i][6].unsqueeze(dim=0))
        for i in pick_neg:
            content.append(self.dataset[tag][0][i][0].unsqueeze(dim=0))
            content_mask.append(self.dataset[tag][0][i][1].unsqueeze(dim=0))
            content_global_attention_mask.append(self.dataset[tag][0][i][2].unsqueeze(dim=0))
            content_emotion.append(self.dataset[tag][0][i][3].unsqueeze(dim=0))
            comments_emotion.append(self.dataset[tag][0][i][4].unsqueeze(dim=0))
            emotion_gap.append(self.dataset[tag][0][i][5].unsqueeze(dim=0))
            style_feature.append(self.dataset[tag][0][i][6].unsqueeze(dim=0))
        content=torch.cat(content,dim=0)
        content_mask=torch.cat(content_mask,dim=0)
        content_global_attention_mask=torch.cat(content_global_attention_mask,dim=0)
        content_emotion=torch.cat(content_emotion,dim=0)
        comments_emotion=torch.cat(comments_emotion,dim=0)
        emotion_gap=torch.cat(emotion_gap,dim=0)
        style_feature=torch.cat(style_feature,dim=0)
        return [content,content_mask,content_global_attention_mask,content_emotion,comments_emotion,emotion_gap,style_feature]

def get_Dd(model,loader,sample_times,category,pool_size,use_cuda):
    
    proportion=[]
    for i in range(0,pool_size+1):
        j=pool_size
        k=i/j
        if k not in proportion:
            proportion.append(k)
    proportion=sorted(proportion)
    Dd={i:{j:[]for j in proportion} for i in category}
    for i in Dd.keys():
        for j in range(0,pool_size+1):
            k=pool_size
            print('category:',i,'num_pos:',j,'num_all:',k)
            
            for l in range(sample_times):
                batch=loader.produce(j,k-j,i)
                batch=data2gpu(batch,use_cuda)
                output=model(i,**batch)
                
                Dd[i][j/k].append(output)
    

    return Dd

def start(i:int):
    seed = 10086
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    config=CDRDConfig()
    model=Generator(config.feature_kernel,config.emb_dim,config.middel_dim,config.W_dim,config.head_num,config.ffn_dim,config.dropout,config.domain_num,config.bert,os.path.join(config.save_param_dir,'semantic_extractor1.pkl'),config.semantic_num,os.path.join(config.save_param_dir,'emotion_extractor1.pkl'),config.emotion_num,os.path.join(config.save_param_dir,'style_extractor1.pkl'),config.style_num,os.path.join(config.save_param_dir,'weight1.pkl'),os.path.join(config.save_param_dir,'crossAttention1.pkl'),os.path.join(config.save_param_dir,'buffer_CDRD1.pkl'),config.language)
    if config.use_cuda:
        model.cuda()
    model.eval()
    if config.language=='cn':
        loader=new_loader(config.max_len,config.bert,config.category_dict,config.data_file+'train_cn.pkl',config.language)
    else:
        loader=new_loader(config.max_len,config.bert,config.category_dict,config.data_file+'train_en.pkl',config.language)
    Dd=get_Dd(model,loader,config.sample_times[i],config.category_dict.values(),config.batch_size,config.use_cuda)
    torch.save(Dd, config.data_file+'Dd.pt')


    
    
        