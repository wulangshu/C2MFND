from run import Run
import numpy as np
import random
import torch
import pandas as pd
from Dd_generator import start


seeds=[2013,2014,2015,2016,2017,2018,2019,2020,2021,2022,2023,2024]
# category_dict = {
#             "科技": 0,  
#             "军事": 1,  
#             "教育考试": 2,  
#             "灾难事故": 3,  
#             "政治": 4,  
#             "医药健康": 5,  
#             "财经商业": 6,  
#             "文体娱乐": 7,  
#             "社会生活": 8
#             }
category_dict = {
            "politifact": 0,  
            "gossipcop": 1,  
            "COVID": 2
            }

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
    k=Run(False)
    result=k.main()
    for i in category_dict.keys():
        tmp['CM_'+i].append(result[i]['fscore'])
    tmp['CM_avg_F1'].append(result['metric'])
    tmp['CM_avg_AUC'].append(result['auc'])
    tmp['CM_avg_Acc'].append(result['acc'])
    tmp['seed'].append(seed)

    pd.DataFrame(tmp).to_excel('En3.xlsx')
pass



