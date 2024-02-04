from run import Run
import numpy as np
import random
import torch
import pandas as pd
from Dd_generator import start
from config import C2MFNDConfig


seeds=[2013,2014,2015,2016,2017,2018,2019,2020,2021,2022,2023,2024]
myconfig=C2MFNDConfig()
language=myconfig.language
category_dict = myconfig.category_dict
tmp={'CM_'+i:[] for i in category_dict.keys()}
tmp['CM_avg_F1']=[]
tmp['CM_avg_AUC']=[]
tmp['CM_avg_Acc']=[]
tmp['seed']=[]

for seed in seeds:
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    k=Run()
    result=k.main()
    for i in category_dict.keys():
        tmp['CM_'+i].append(result[i]['fscore'])
    tmp['CM_avg_F1'].append(result['metric'])
    tmp['CM_avg_AUC'].append(result['auc'])
    tmp['CM_avg_Acc'].append(result['acc'])
    tmp['seed'].append(seed)
    if language=='cn':
        pd.DataFrame(tmp).to_excel('Weibo21_'+str(i)+'.xlsx')
    else:
        pd.DataFrame(tmp).to_excel('En3_'+str(i)+'.xlsx')
pass



