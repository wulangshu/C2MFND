import torch 
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self,dim_input,dim_output):
        super(Attention,self).__init__()
        self.dim_input=dim_input
        self.dim_output=dim_output
        self.Q=nn.Parameter(torch.empty((self.dim_input,self.dim_output)))
        self.K=nn.Parameter(torch.empty((self.dim_input,self.dim_output)))
        self.V=nn.Parameter(torch.empty((self.dim_input,self.dim_output)))
        
        self.dropout = nn.Dropout(p=0.2)############

        nn.init.kaiming_uniform_(self.Q)
        nn.init.kaiming_uniform_(self.K)
        nn.init.kaiming_uniform_(self.V)

    def forward(self,features_host,features_guests):
        
        query=torch.matmul(features_host,self.Q)
        key=torch.matmul(features_guests,self.K)
        value=torch.matmul(features_guests,self.V)
        att=torch.matmul(query,key.T)/torch.sqrt(torch.tensor(self.dim_output))
        attention=torch.softmax(att,dim=1)
        attention = self.dropout(attention)
        return torch.matmul(attention,value)

class MutilAttention(nn.Module):
    def __init__(self,head_num,dim_input,dim_output,ffn_dim) -> None:
        super(MutilAttention,self).__init__()
        
        self.mutilhead=nn.ModuleList([Attention(dim_input,dim_output) for i in range(head_num)])
        self.W=nn.Linear(dim_output*head_num,dim_input)
        self.Norm1=nn.LayerNorm(dim_input,elementwise_affine=False)
        Ffn=[]
        Ffn.append(nn.Linear(dim_input,ffn_dim*dim_input))
        Ffn.append(nn.ReLU())
        Ffn.append(nn.Dropout(p=0.2))
        Ffn.append(nn.Linear(ffn_dim*dim_input,dim_input))
        self.Ffn=nn.Sequential(*Ffn)
        self.Norm2=nn.LayerNorm(dim_input,elementwise_affine=False)
        
    def forward(self,features_host,features_guests):
        
        feature_input=features_host
        feature=[]
        for i in self.mutilhead:
            feature.append(i(features_host,features_guests))
        feature=torch.cat(feature,dim=1)
        feature=self.W(feature)
        feature+=feature_input
        feature=self.Norm1(feature)

        feature_input=feature
        feature=self.Ffn(feature)
        feature+=feature_input
        feature=self.Norm2(feature)
        return feature
    
class CrossAttention(nn.Module):
    def __init__(self,dim_input,dim_output,head_num,ffn_dim):
        super(CrossAttention,self).__init__()
        # self.pos_embedding=nn.Parameter(torch.empty((1,dim_input),device=device))
        # nn.init.kaiming_uniform_(self.pos_embedding)
        self.sentence_extractor=MutilAttention(head_num,dim_input,dim_output,ffn_dim)

    def forward(self,features_host,features_guests):
        # pos_host=torch.cat([self.pos_embedding for i in range(features_host.shape[0])],dim=0)
        # pos_guests=torch.cat([self.pos_embedding for i in range(features_guests.shape[0])],dim=0)
        return self.sentence_extractor(features_host,features_guests)
    
    


class Disbiaser(nn.Module):
    def __init__(self,dim_input,dim_output):
        super(Disbiaser,self).__init__()
        self.dim_input=dim_input
        self.dim_output=dim_output
        self.Q=nn.Parameter(torch.empty((self.dim_input,self.dim_output)))
        self.K=nn.Parameter(torch.empty((self.dim_input,self.dim_output)))
        nn.init.kaiming_uniform_(self.Q)
        nn.init.kaiming_uniform_(self.K)

    def forward(self,features_host,features_guests):
        
        query=torch.matmul(features_host,self.Q)
        key=torch.matmul(features_guests,self.K)
        attention=torch.softmax(torch.matmul(query,key.T)/torch.sqrt(torch.tensor(self.dim_output)),dim=1)
        return attention


class MLP(nn.Module):
    def __init__(self,dim_input,dim_embs,dropout, output_layer=False):
        super(MLP,self).__init__()
        layers=list()
        for embed_dim in dim_embs:
            layers.append(nn.Linear(dim_input,embed_dim))
            layers.append(torch.nn.BatchNorm1d(embed_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout))
            dim_input=embed_dim
        if output_layer:
            layers.append(torch.nn.Linear(embed_dim, 1))
        self.layers=nn.Sequential(*layers)

    def forward(self,features_input):
        return self.layers(features_input)
        

class Cnn_extractor(nn.Module):
    def __init__(self, feature_kernel, input_size):
        super(Cnn_extractor, self).__init__()
        self.convs = torch.nn.ModuleList(
            [torch.nn.Conv1d(input_size, feature_num, kernel)
             for kernel, feature_num in feature_kernel.items()])
        input_shape = sum([feature_kernel[kernel] for kernel in feature_kernel])

    def forward(self, input_data):
        share_input_data = input_data.permute(0, 2, 1)
        feature = [conv(share_input_data) for conv in self.convs]
        feature = [torch.max_pool1d(f, f.shape[-1]) for f in feature]
        feature = torch.cat(feature, dim=1)
        feature = feature.view([-1, feature.shape[1]])
        return feature
