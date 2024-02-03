import torch
from copy import deepcopy
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score, roc_auc_score
import numpy as np
import math

class Averager():

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v
    

def data2gpu(batch, use_cuda):
    if use_cuda:
        batch_data = {
            'content': batch[0].cuda(),
            'content_masks': batch[1].cuda(),
            'content_global_attention_mask': batch[2].cuda(),
            'content_emotion': batch[3].cuda(),
            'comments_emotion': batch[4].cuda(),
            'emotion_gap': batch[5].cuda(),
            'style_feature': batch[6].cuda(),
            'label': batch[7].cuda(),
            'category': batch[8].cuda()
            }
    else:
        batch_data = {
            'content': batch[0],
            'content_masks': batch[1],
            'content_global_attention_mask': batch[2],
            'content_emotion': batch[3],
            'comments_emotion': batch[4],
            'emotion_gap': batch[5],
            'style_feature': batch[6],
            'label': batch[7],
            'category': batch[8]
            }
    return batch_data

class Recorder():
    def __init__(self,early_stop):
        self.early_stop=early_stop
        self.max={'metric':0}
        self.cur={'metric':0}
        self.max_index=0
        self.cur_index=0

    def add(self,x):
        self.cur=x
        self.cur_index+=1
        print("curent",self.cur)
        return self.judge()
    
    def judge(self):
        if self.max['metric']<self.cur['metric']:
            self.max=self.cur
            self.max_index=self.cur_index
            self.showfinal()
            return 'save'
        self.showfinal()
        if self.cur_index-self.max_index>self.early_stop:
            return 'esc'
        else:
            return 'continue'
        
    def showfinal(self):
        print("Max",self.max)
        

class Buffer():
    def __init__(self,domain_num,pool_size,dim,device='cuda'):
        self.device=device
        self.pool={i:torch.empty((pool_size,dim),device=self.device).float() for i in range(domain_num)}
        for i in range(domain_num):
            torch.nn.init.kaiming_uniform_(self.pool[i])
        self.pool_size=pool_size
        self.dim=dim
    
    def update(self,content,tag):
        self.pool[tag]=content

    
def getfeature_domain(buffer:Buffer,tag=None):
    if tag!=None:
        return buffer.pool[tag]
       
    return buffer.pool
    
    

def metrics(y_true, y_pred, category, category_dict):
    res_by_category = {}
    metrics_by_category = {}
    reverse_category_dict = {}
    for k, v in category_dict.items():
        reverse_category_dict[v] = k
        res_by_category[k] = {"y_true": [], "y_pred": []}

    for i, c in enumerate(category):
        c = reverse_category_dict[c]
        res_by_category[c]['y_true'].append(y_true[i])
        res_by_category[c]['y_pred'].append(y_pred[i])

    for c, res in res_by_category.items():
        try:
            metrics_by_category[c] = {
                'auc': roc_auc_score(res['y_true'], res['y_pred']).round(4).tolist()
            }
        except Exception as e:
            metrics_by_category[c] = {
                'auc': 0
            }

    metrics_by_category['auc'] = roc_auc_score(y_true, y_pred, average='macro')
    y_pred = np.around(np.array(y_pred)).astype(int)
    metrics_by_category['metric'] = f1_score(y_true, y_pred, average='macro')
    metrics_by_category['recall'] = recall_score(y_true, y_pred, average='macro')
    metrics_by_category['precision'] = precision_score(y_true, y_pred, average='macro')
    metrics_by_category['acc'] = accuracy_score(y_true, y_pred)
    
    for c, res in res_by_category.items():
        try:
            metrics_by_category[c] = {
                'precision': precision_score(res['y_true'], np.around(np.array(res['y_pred'])).astype(int), average='macro').round(4).tolist(),
                'recall': recall_score(res['y_true'], np.around(np.array(res['y_pred'])).astype(int), average='macro').round(4).tolist(),
                'fscore': f1_score(res['y_true'], np.around(np.array(res['y_pred'])).astype(int), average='macro').round(4).tolist(),
                'auc': metrics_by_category[c]['auc'],
                'acc': accuracy_score(res['y_true'], np.around(np.array(res['y_pred'])).astype(int)).round(4)
            }
        except Exception as e:
            metrics_by_category[c] = {
                'precision': 0,
                'recall': 0,
                'fscore': 0,
                'auc': 0,
                'acc': 0
            }
    return metrics_by_category

def experiment(proportion,domain_num,batch_size):
    simulation={i:{(j/batch_size):0 for j in range(batch_size+1)}for i in range(domain_num)}
    for i in range(domain_num):
        for j in range(batch_size+1):
            simulation[i][j/batch_size]=math.comb(batch_size,j)*math.pow(proportion[i],j)*math.pow(1-proportion[i],batch_size-j)
    return simulation

def get_D_feature(Dd,simulation,tmp,domain_disbiaser):
    box=[]
    for i in simulation.keys():
        tempy=[]
        for dd in Dd[i]:
            attention=domain_disbiaser(tmp,dd)
            tempy.append(torch.matmul(attention,dd))
        tempy=torch.mean(torch.cat(tempy,dim=0),dim=0).unsqueeze(dim=0)
        box.append(tempy)
    box=torch.cat(box,dim=0)
    simulation_new=torch.tensor(list(simulation.values()),device=tmp.device).unsqueeze(dim=0)
    output=torch.matmul(simulation_new,box)

    return output
                    