# C2MFND
Access to two datasets and Baselines:

Weibo21: https://github.com/kennqiang/MDFEND-Weibo21

En3 and Baselines: https://github.com/ICTMCG/M3FEND

Access to used pretrained Robertas:

Chinese: https://huggingface.co/hfl/chinese-roberta-wwm-ext

English: https://huggingface.co/FacebookAI/roberta-base


Hyperparameters & Values ： 

epoches   & 50        \\ 
early stop   & 3 (w/o CI), 10 (w/ CI)        \\ 
$n_{bat}$  & 64        \\  
$n_w$ & 170 (Weibo21), 512 (En3) \\  
optimizer & Adam \\  
learning rate & 5e-4 \\  
dropout & 0.2 \\  
$d_j$ & 768 \\  
kernels for TextCNN & $\{1,2,3,5,10\}$ \\  
$n_u$ & 8 \\
$d_z$ & 320 \\
$n_{head}$ & 4\\
$d_k$ & 240 \\
$FFN dimension$ & 320 \\
$n_{freq}$ & 3 \\
$MLP dimension$ & 384\\
