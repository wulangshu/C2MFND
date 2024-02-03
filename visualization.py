#使用t-SNE降维
from sklearn import manifold
from CDRD import CrossDomainRDModel
import torch 
import torch.nn as nn
import os
import tqdm
import numpy as np
import pickle
from config import CDRDConfig
from utils import data2gpu,experiment
from layer import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from dataloader import bert_data
from PIL import Image, ImageDraw, ImageFont


class Visualizer():
    def __init__(self,
                 config,
                 is_causal
                 ):
        self.lr = config.lr
        self.early_stop=config.early_stop
        self.weight_decay = config.weight_decay
        
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
        
        

        loader=bert_data(max_len = config.max_len, batch_size = config.batch_size, pool_size=config.pool_size, bert = config.bert,
                        category_dict = config.category_dict,category_count=config.category_count,language=config.language)
        if config.language=='cn':
            self.train_loader,train_proportion=loader.load_data(config.data_file+'train_cn.pkl',True)
            self.test_loader,test_proportion=loader.load_data(config.data_file+'test_cn.pkl',True)
            self.category_dict_res={
                0:"Science",
                1:"Military",
                2:"Education",
                3:"Disaster",
                4:"Politics",
                5:"Health",
                6:"Finance",
                7:"Entertainment",
                8:"Society"
                }
        else:
            self.train_loader,train_proportion=loader.load_data(config.data_file+'train_en.pkl',True)
            self.test_loader,test_proportion=loader.load_data(config.data_file+'test_en.pkl',True)
            self.category_dict_res={i:list(self.category_dict.keys())[i] for i in list(self.category_dict.values())}
        
        self.is_causal=is_causal
        if self.is_causal:
            self.train_buffer_simulation=experiment(train_proportion,config.domain_num,config.batch_size)
        else:
            self.train_buffer_simulation=None

        if not os.path.exists(config.save_param_dir):
            self.save_param_dir = os.makedirs(config.save_param_dir)
        else:
            self.save_param_dir = config.save_param_dir
        
        
        
    def start(self,fig1,fig2,seed):
        

        self.model=CrossDomainRDModel(self.feature_kernel,self.domain_num,self.emb_dim,self.mlp_dim,self.W_dim,self.head_num,self.ffn_dim,self.bert,self.Dd,self.pool_size,self.middel_dim,self.dropout,self.is_causal,self.language,self.semantic_num,self.emotion_num,self.style_num)

        if self.use_cuda:
            self.model.cuda()
            
        if self.is_causal:
            self.model.load_state_dict(torch.load(os.path.join(self.save_param_dir,'parameter_CDRD_withcausal.pkl')))
            with open(os.path.join(self.save_param_dir,'buffer_CDRD_withcausal.pkl'), 'rb') as file:
                self.model.buffer = pickle.load(file)
        else:
            self.model.load_state_dict(torch.load(os.path.join(self.save_param_dir,'parameter_CDRD_nocausal.pkl')))
            with open(os.path.join(self.save_param_dir,'buffer_CDRD_nocausal.pkl'), 'rb') as file:
                self.model.buffer = pickle.load(file)

        out_F,out_T = [],[]
        label = []
        category = []
        self.model.eval()
        data_iter = tqdm.tqdm(self.test_loader)
        for step_n, batch in enumerate(data_iter):
            with torch.no_grad():
                batch_data = data2gpu(batch, self.use_cuda)
                batch_label = batch_data['label']
                batch_category = batch_data['category']
                _,F_output,T_output = self.model(self.is_causal,self.train_buffer_simulation,**batch_data)

                label.extend(batch_label.detach().cpu().numpy().tolist())
                category.extend(batch_category.detach().cpu().numpy().tolist())
                out_F.extend(F_output.detach().cpu().numpy().tolist())
                out_T.extend(T_output.detach().cpu().numpy().tolist())
        
        out_F,out_T=np.array(out_F),np.array(out_T)
        ts_F,ts_T=manifold.TSNE(n_components=2,random_state=seed),manifold.TSNE(n_components=2,random_state=seed)
        out_redu_F,out_redu_T=ts_F.fit_transform(out_F),ts_T.fit_transform(out_T)
        
        color_positive=['navy','darkorange','darkgreen','darkred','purple','saddlebrown','deeppink','dimgray','goldenrod','black']
        color_projection_positive={i:color_positive[i] for i in range(self.domain_num)}
        color_negative=['blue','orange','limegreen','red','fuchsia','chocolate','hotpink','slategray','gold','black']
        color_projection_negative={i:color_negative[i] for i in range(self.domain_num)}
        

        category=[[category[i],label[i]] for i in range(len(category))]
        category=np.array(list(map(lambda x: color_projection_positive[x[0]] if x[1] else color_projection_negative[x[0]],category)))
        label=np.array(label)
        pos_index,neg_index=label==1,label==0
        category_pos,category_neg=category[pos_index],category[neg_index]
    
       
       
        out_redu_pos,out_redu_neg=out_redu_F[pos_index],out_redu_F[neg_index]
        
        if self.is_causal:
            ax=fig1.add_subplot(133)
            ax.axis('off')
            ax.set_title('(c) L',fontdict={'family':'Times New Roman','size':48},loc='center',y=-0.1) # r'$\mathcal{L}$'
        else:
            ax=fig1.add_subplot(132)
            ax.axis('off')
            ax.set_title('(b) F',fontdict={'family':'Times New Roman','size':48},loc='center',y=-0.1)
        pos_sca=ax.scatter(out_redu_pos[:,0],out_redu_pos[:,1],c=category_pos,marker='^',s=30,alpha=0.6,label='Positive')
        neg_sca=ax.scatter(out_redu_neg[:,0],out_redu_neg[:,1],c=category_neg,marker='v',s=30,alpha=0.6,label='Negative')
              
        
        
        out_redu_pos,out_redu_neg=out_redu_T[pos_index],out_redu_T[neg_index]
        if not self.is_causal:
            ax=fig1.add_subplot(131)
            ax.axis('off')
            ax.set_title('(a) T',fontdict={'family':'Times New Roman','size':48},loc='center',y=-0.1)
            pos_sca=ax.scatter(out_redu_pos[:,0],out_redu_pos[:,1],c=category_pos,marker='^',s=30,alpha=0.6,label='Positive')
            neg_sca=ax.scatter(out_redu_neg[:,0],out_redu_neg[:,1],c=category_neg,marker='v',s=30,alpha=0.6,label='Negative')
        
        
        
        sp=2
        color_positive,color_negative=color_projection_positive[sp],color_projection_negative[sp]
        sp_index=(category==color_positive)|(category==color_negative)
        sp_category,sp_label=category[sp_index],label[sp_index]
        sp_out_redu_F=out_redu_F[sp_index]
        pos_index,neg_index=sp_label==1,sp_label==0
        out_redu_pos,out_redu_neg=sp_out_redu_F[pos_index],sp_out_redu_F[neg_index]

        if self.is_causal:
            ax=fig2.add_subplot(122)
            ax.axis('off')
            ax.set_title('(b) w/ CI',fontdict={'family':'Times New Roman','size':48},loc='center',y=-0.1)
        else:
            ax=fig2.add_subplot(121)
            ax.axis('off')
            ax.set_title('(a) w/o CI',fontdict={'family':'Times New Roman','size':48},loc='center',y=-0.1)
        pos_sca=ax.scatter(out_redu_pos[:,0],out_redu_pos[:,1],c='darkred',marker='^',s=50,alpha=0.6,label='Positive')
        neg_sca=ax.scatter(out_redu_neg[:,0],out_redu_neg[:,1],c='navy',marker='v',s=50,alpha=0.6,label='Negative')
        
           
        

fig1,fig2=plt.figure(figsize=(24,8)),plt.figure(figsize=(16,8))
# domain_num=9
# category_dict_res={
#                 0:"Science",
#                 1:"Military",
#                 2:"Education",
#                 3:"Disaster",
#                 4:"Politics",
#                 5:"Health",
#                 6:"Finance",
#                 7:"Entertainment",
#                 8:"Society"
#                 }

domain_num=3
category_dict_res={
            0:"politifact",  
            1:"gossipcop",  
            2:"COVID"
            }
color_positive=['navy','darkorange','darkgreen','darkred','purple','saddlebrown','deeppink','dimgray','goldenrod','black']
color_projection_positive={i:color_positive[i] for i in range(domain_num)}
color_negative=['blue','orange','limegreen','red','fuchsia','chocolate','hotpink','slategray','gold','black']
color_projection_negative={i:color_negative[i] for i in range(domain_num)}
color_handles_positive=[mlines.Line2D([0], [0], color='white',markeredgecolor =color_projection_positive[i],markerfacecolor=color_projection_positive[i],markersize=16, marker='^', label=category_dict_res[i]) for i in range(domain_num)]
color_handles_negative=[mlines.Line2D([0], [0], color='white',markeredgecolor =color_projection_negative[i],markerfacecolor=color_projection_negative[i],markersize=16, marker='v', label=category_dict_res[i]) for i in range(domain_num)]
handles_fig2=[mlines.Line2D([0], [0], color='white',markerfacecolor= 'darkred',markersize=32, marker='^', label='Positive'),
         mlines.Line2D([0], [0], color='white',markerfacecolor= 'navy',markersize=32, marker='v', label='Negative')
         ]

seed=2014
Visualizer(CDRDConfig(),False).start(fig1,fig2,seed)
fig1.subplots_adjust(wspace=0)
l1=fig1.legend(prop = {'family': 'Times New Roman','size':16},handles=color_handles_positive,bbox_to_anchor=(0.97,0.95),frameon=True,title='Positive',title_fontproperties={'family': 'Times New Roman','size':16},handletextpad=0)
fig1.legend(prop = {'family': 'Times New Roman','size':16},handles=color_handles_negative,bbox_to_anchor=(0.97,0.5),frameon=True,title='Negative',title_fontproperties={'family': 'Times New Roman','size':16},handletextpad=0)
fig1.add_artist(l1)

Visualizer(CDRDConfig(),True).start(fig1,fig2,seed)
fig2.subplots_adjust(wspace=0)
fig2.legend(prop = {'family': 'Times New Roman','size':36},handles=handles_fig2,bbox_to_anchor=(0.97,0.95),frameon=True,handletextpad=0)

fig1.savefig('distribution'+str(seed)+'.pdf', bbox_inches='tight')
fig2.savefig('CI_Ablation'+str(seed)+'.pdf', bbox_inches='tight')
# fig1.savefig('distribution'+str(seed)+'.pdf')
# fig2.savefig('CI_Ablation'+str(seed)+'.pdf')
pass
          