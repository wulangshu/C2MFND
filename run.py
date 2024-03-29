import torch 
import os
from dataloader import bert_data
from C2MFND import Trainer as C2MFNDtrainer
from config import C2MFNDConfig
from utils import experiment


class Run():
    def __init__(self):
        self.config=C2MFNDConfig()


    def get_dataloader(self):
        loader=bert_data(max_len = self.config.max_len, batch_size = self.config.batch_size, pool_size=self.config.pool_size, bert = self.config.bert,
                        category_dict = self.config.category_dict,category_count=self.config.category_count,language=self.config.language)
        if self.config.language=='cn':
            train_loader,train_proportion=loader.load_data(self.config.data_file+'train_cn.pkl',True)
            val_loader,val_proportion=loader.load_data(self.config.data_file+'val_cn.pkl',True)#
            test_loader,test_proportion=loader.load_data(self.config.data_file+'test_cn.pkl',True)#
        else:
            train_loader,train_proportion=loader.load_data(self.config.data_file+'train_en.pkl',True)
            val_loader,val_proportion=loader.load_data(self.config.data_file+'val_en.pkl',True)#
            test_loader,test_proportion=loader.load_data(self.config.data_file+'test_en.pkl',True)#
        return train_loader,train_proportion,val_loader,val_proportion,test_loader,test_proportion

    def main(self):
        train_loader,train_proportion,val_loader,val_proportion,test_loader,test_proportion=self.get_dataloader()#使用proportion检测发现模拟实验与数据集分布有一致性，且呈对数正态分布
        if self.config.model_name=='C2MFND':
            trainer=C2MFNDtrainer(self.config,train_loader,val_loader,test_loader,False)
        if self.config.model_name=='C2MFND_causal':
            train_buffer_simulation,val_buffer_simulation,test_buffer_simulation=experiment(train_proportion,self.config.domain_num,self.config.batch_size),experiment(val_proportion,self.config.domain_num,self.config.batch_size),experiment(test_proportion,self.config.domain_num,self.config.batch_size)
            trainer=C2MFNDtrainer(self.config,train_loader,val_loader,test_loader,True,train_buffer_simulation,val_buffer_simulation,test_buffer_simulation)
        
        k,_=trainer.train()
        return k
        