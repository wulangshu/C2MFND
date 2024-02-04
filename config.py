import torch
class C2MFNDConfig():
    def __init__(self):
        self.model_name="C2MFND"
        self.model_name="C2MFND_causal"
        # self.language='en_former'
        self.language='en'
        # self.language='cn'
        
        self.root_dir="."
        if self.model_name=='C2MFND':
            self.early_stop=3
        else:
            self.early_stop=10

        self.dropout=0.2
        self.weight_decay=5e-5
        self.epoches=50
        self.batch_size = 64#
        
        self.emb_dim=768
          
        self.middel_dim = 320
        self.mlp_dim= 384
        self.head_num= 4
        self.W_dim=0.75
        self.ffn_dim=1
        
        self.pool_size=8
        
        self.semantic_num=7
        self.emotion_num=8
        self.style_num=2

        self.sample_times=3
        self.feature_kernel={1: 64, 2: 64, 3: 64, 5: 64, 10: 64}
        if self.language=='cn':
            self.lr=0.0005
            self.max_len=170
            self.bert=self.root_dir+'/chinese_roberta_wwm_ext'
            self.category_dict = {
            "科技": 0,  
            "军事": 1,  
            "教育考试": 2,  
            "灾难事故": 3,  
            "政治": 4,  
            "医药健康": 5,  
            "财经商业": 6,  
            "文体娱乐": 7,  
            "社会生活": 8
            }
            self.category_count={'科技':0,'军事':0,'教育考试':0,'灾难事故':0,'政治':0,'医药健康':0,'财经商业':0,'文体娱乐':0,'社会生活':0}
            self.domain_num=9
  
        elif self.language=='en':
            self.lr=0.0001
            self.max_len=512
            self.bert=self.root_dir+'/english_roberta_base'
            self.category_dict = {
            "politifact": 0,  
            "gossipcop": 1,  
            "COVID": 2
            }
            self.category_count={"politifact":0,"gossipcop":0,"COVID":0}
            self.domain_num=3
        else:
            self.lr=0.0001
            self.max_len=4096
            self.bert=self.root_dir+'/longformer'
            self.category_dict = {
            "politifact": 0,  
            "gossipcop": 1,  
            "COVID": 2
            }
            self.category_count={"politifact":0,"gossipcop":0,"COVID":0}
            self.domain_num=3
            

        self.data_file= self.root_dir+'/data/'
        self.save_param_dir=self.root_dir+'/param_model/'
        

        if torch.cuda.is_available()==True:
            self.use_cuda=True
        else:
            self.use_cuda=False
        
