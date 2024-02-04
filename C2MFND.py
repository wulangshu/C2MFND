import torch 
import torch.nn as nn
import os
import tqdm
import numpy as np
import pickle
from transformers import BertModel,RobertaModel,LongformerModel
from utils import data2gpu,Averager,metrics,Recorder,Buffer,getfeature_domain,get_D_feature
from layer import *

class CrossDomainRDModel(nn.Module):
    def __init__(self,feature_kernel,domain_num,emb_dim,mlp_dim,W_dim,head_num,ffn_dim,bert,Dd,pool_size,middel_dim,dropout,is_causal,language,semantic_num,emotion_num,style_num):
        super(CrossDomainRDModel,self).__init__()
        self.domain_num=domain_num
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
        self.buffer=Buffer(domain_num,pool_size,self.mid_dim)
        
        self.semantic_extractor=nn.ModuleList([Cnn_extractor(feature_kernel,emb_dim) for i in range(self.semantic_num)])
        if language=='cn':
            self.emotion_extractor=nn.ModuleList([MLP(47*5,[int(0.8*middel_dim),middel_dim],dropout) for i in range(self.emotion_num)])
            self.style_extractor = nn.ModuleList([MLP(48,[int(0.8*middel_dim),middel_dim],dropout) for i in range(self.style_num)])
        else:
            self.emotion_extractor=nn.ModuleList([MLP(38*5,[int(0.8*middel_dim),middel_dim],dropout) for i in range(self.emotion_num)])
            self.style_extractor = nn.ModuleList([MLP(32,[int(0.8*middel_dim),middel_dim],dropout) for i in range(self.style_num)])

    
        self.weight=nn.Parameter(torch.full((1,1,(self.semantic_num+self.emotion_num+self.style_num)),1/(self.semantic_num+self.emotion_num+self.style_num)))
        

        self.W_f=nn.Linear(self.mid_dim,self.mid_dim)
        self.W_d=nn.Linear(self.mid_dim,self.mid_dim)
        self.crossattention=CrossAttention(self.mid_dim,int(W_dim*self.mid_dim),head_num,ffn_dim)
        self.rumourMLPforT=MLP(self.mid_dim,[mlp_dim],dropout,output_layer=True)
        self.rumourMLPforF=MLP(self.mid_dim,[mlp_dim],dropout,output_layer=True)
        

        if is_causal==True:
            self.domain_disbiaser=Disbiaser(middel_dim,int(W_dim*self.mid_dim))
            self.Dd=torch.load(Dd+'Dd.pt')

   
    def forward(self,is_causal,simulation,**kwargs):
        inputs = kwargs['content']
        masks = kwargs['content_masks']
        global_attention_mask = kwargs['content_global_attention_mask']
        category = kwargs['category']
        
        
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

        feature_domain=getfeature_domain(self.buffer)

        F_feature=[]
        if is_causal==True:
            D_feature=[]
        for i in range(len(category)):
            tmp=self.crossattention(T_feature[i].unsqueeze(dim=0),feature_domain[category[i].item()])
            F_feature.append(tmp)
            if is_causal==True:
                tmp=get_D_feature(self.Dd[category[i].item()],simulation[category[i].item()],tmp,self.domain_disbiaser)
                D_feature.append(tmp)

        F_feature=torch.cat(F_feature,dim=0)
        if is_causal==True:
            D_feature=torch.cat(D_feature)
      
        if is_causal==True:   
            L_feature=self.W_f(F_feature)+self.W_d(D_feature)
        else:
            L_feature=self.W_f(F_feature)

        pred_F=self.rumourMLPforF(L_feature)
        pred_T=self.rumourMLPforT(T_feature)
        
        pred_final=torch.sigmoid(pred_F.squeeze(1)+pred_T.squeeze(1))

        return pred_final,L_feature,T_feature

    
    
    def write(self,**kwargs):
        inputs = kwargs['content']
        masks = kwargs['content_masks']
        global_attention_mask = kwargs['content_global_attention_mask']
        category = kwargs['category']
        
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
        
        for i in range(self.domain_num):
            domain_old=getfeature_domain(self.buffer,tag=i)
            T_new=T_feature[(category==i)]
            if T_new.shape[0]!=0:
                domain_new=self.crossattention(domain_old.detach(),T_new.detach())
                self.buffer.update(domain_new,i)


            
class Trainer():
    def __init__(self,
                 config,
                 train_loader,
                 val_loader,
                 test_loader,
                 is_causal,
                 train_buffer_simulation=None,
                 val_buffer_simulation=None,
                 test_buffer_simulation=None
                 ):
        self.lr = config.lr
        self.early_stop=config.early_stop
        self.weight_decay = config.weight_decay
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.val_loader = val_loader
        self.epoches = config.epoches
        self.category_dict = config.category_dict
        self.use_cuda = config.use_cuda
        self.emb_dim = config.emb_dim
        self.mlp_dim = config.mlp_dim
        self.domain_num=config.domain_num
        self.bert = config.bert
        self.Dd = config.data_file
        self.dropout = config.dropout
        self.W_dim=config.W_dim
        self.head_num=config.head_num
        self.ffn_dim=config.ffn_dim
        self.pool_size=config.pool_size
        self.middel_dim=config.middel_dim 
        self.feature_kernel=config.feature_kernel
        self.language=config.language
        self.semantic_num=config.semantic_num
        self.emotion_num=config.emotion_num
        self.style_num=config.style_num

        self.is_causal=is_causal
        self.train_buffer_simulation=train_buffer_simulation
        self.val_buffer_simulation=val_buffer_simulation
        self.test_buffer_simulation=test_buffer_simulation
        
        
        if not os.path.exists(config.save_param_dir):
            self.save_param_dir = os.makedirs(config.save_param_dir)
        else:
            self.save_param_dir = config.save_param_dir
        
    def train(self):
        
        self.model=CrossDomainRDModel(self.feature_kernel,self.domain_num,self.emb_dim,self.mlp_dim,self.W_dim,self.head_num,self.ffn_dim,self.bert,self.Dd,self.pool_size,self.middel_dim,self.dropout,self.is_causal,self.language,self.semantic_num,self.emotion_num,self.style_num)

        if self.use_cuda:
            self.model.cuda()
        loss_fn = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        recorder=Recorder(self.early_stop)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=100,gamma=0.98)
         
        for epoch in range(self.epoches):
            self.model.train()
            train_data_iter =tqdm.tqdm(self.train_loader)
            avg_loss=Averager()
            
            for num_n,batch in enumerate(train_data_iter):
                batch_data=data2gpu(batch,self.use_cuda)
                label= batch_data['label']
                category = batch_data['category']
                
                label_pred,*_=self.model(self.is_causal,self.train_buffer_simulation,**batch_data)
                loss =loss_fn(label_pred, label.float())
                loss.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                avg_loss.add(loss.item())

                optimizer.zero_grad()
                self.model.write(**batch_data)

                      
            print('Training Epoch {};Loss {}; '.format(epoch+1,avg_loss.item()))

            results =self.test(self.is_causal,self.train_buffer_simulation,self.val_loader)
            
            mark =recorder.add(results)

            if mark=='save':
                torch.save(self.model.state_dict(),os.path.join(self.save_param_dir,'parameter_C2MFND.pkl'))
                with open(os.path.join(self.save_param_dir,'buffer_C2MFND.pkl'), 'wb') as file:
                    pickle.dump(self.model.buffer, file)
            elif mark=='esc':
                break
            else:
                continue
            
        self.model.load_state_dict(torch.load(os.path.join(self.save_param_dir,'parameter_C2MFND.pkl')))
        with open(os.path.join(self.save_param_dir,'buffer_C2MFND.pkl'), 'rb') as file:
            self.model.buffer = pickle.load(file)

        results = self.test(self.is_causal,self.train_buffer_simulation,self.test_loader)
        print(results)

        if self.is_causal==False:
            semantic_extractor=self.model.semantic_extractor
            torch.save(semantic_extractor.state_dict(),os.path.join(self.save_param_dir,'semantic_extractor.pkl'))
            emotion_extractor=self.model.emotion_extractor
            torch.save(emotion_extractor.state_dict(),os.path.join(self.save_param_dir,'emotion_extractor.pkl'))
            style_extractor=self.model.style_extractor
            torch.save(style_extractor.state_dict(),os.path.join(self.save_param_dir,'style_extractor.pkl'))
            weight=self.model.weight
            torch.save(weight,os.path.join(self.save_param_dir,'weight.pkl'))
            crossAttention=self.model.crossattention
            torch.save(crossAttention.state_dict(),os.path.join(self.save_param_dir,'crossAttention.pkl'))
            

        return results, os.path.join(self.save_param_dir,'parameter_C2MFND.pkl')
            
    def test(self,is_causal,simulation,dataloader):
        pred = []
        label = []
        category = []
        self.model.eval()
        data_iter = tqdm.tqdm(dataloader)
        for step_n, batch in enumerate(data_iter):
            with torch.no_grad():
                batch_data = data2gpu(batch, self.use_cuda)
                batch_label = batch_data['label']
                batch_category = batch_data['category']
                batch_label_pred,*_ = self.model(is_causal,simulation,**batch_data)

                label.extend(batch_label.detach().cpu().numpy().tolist())
                pred.extend(batch_label_pred.detach().cpu().numpy().tolist())
                category.extend(batch_category.detach().cpu().numpy().tolist())
        
        return metrics(label, pred, category, self.category_dict)
