import pandas as pd
import os
import stylecloud
from PIL import Image, ImageDraw, ImageFont
import json
import requests
import time
import pickle
import numpy as np

apikey="#"
def translate(sentence, src_lan, tgt_lan):
    url = 'http://api.niutrans.com/NiuTransServer/translation?'
    
    data = {"from": src_lan, "to": tgt_lan, "apikey": apikey, "src_text": sentence}
    res = requests.post(url, data = data)
    res_dict = json.loads(res.text)
    if "tgt_text" in res_dict:
        result = res_dict['tgt_text']
    else:
        result = res
    return result


domain_num=9
domain_dict_reverse={
            "科技": 0,
            "军事": 1,
            "教育考试": 2,
            "灾难事故": 3,
            "政治": 4,
            "医药健康": 5,
            "财经商业": 6,
            "文体娱乐": 7,
            "社会生活": 8,
        }
domain_dict_en={
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

keys=list(domain_dict_reverse.keys())
domain_dict ={i:keys[i] for i in range(domain_num)}

text_domain={i:[] for i in range(domain_num)}

with open('./data/stop_words.txt','r') as file:
    stopwords = file.read().split('\n')

# for path in ['./data/train_cn.pkl','./data/val_cn.pkl','./data/test_cn.pkl']:
#     file=pd.read_pickle(path)
#     for i in range(domain_num-1,-1,-1):
#         text_all_ch=list(file.loc[file['category']== domain_dict[i],'content'])
#         text_all=[]
#         for text in text_all_ch:
#             loc=0
            
#             while loc<=len(text):
#                 if loc+4995<=len(text):

#                     try:
#                         trans = translate(text[loc:loc+4995], 'auto', 'en')
#                         if type(trans) in [str]:
#                             text_all.append(trans)
#                         else:
#                             pass
#                     except Exception as exc:
#                         print(exc)
#                 else:
#                     try:
#                         trans = translate(text[loc:], 'auto', 'en')
#                         if type(trans) in [str]:
#                             text_all.append(trans)
#                         else:
#                             pass
#                     except Exception as exc:
#                         print(exc)
                
#                 time.sleep(0.205)
                
#                 loc+=4995
            
#         with open(path+'text_en_domain_'+str(i)+'.pkl', 'wb') as dump_file:
#             pickle.dump(text_all, dump_file)

for path in ['./data/train_cn.pkl','./data/val_cn.pkl','./data/test_cn.pkl']:
    for i in range(domain_num-1,-1,-1):
        with open(path+'text_en_domain_'+str(i)+'.pkl', 'rb') as read_file:
            temp=pickle.load(read_file)
            text_domain[i].extend(temp)

for i in range(domain_num-1,-1,-1):
    text_domain[i]=' '.join(text_domain[i])
    

image_all=Image.new("RGB", size=(512*3, 316*3))
x,y=0,0

bg_fig=Image.open('./data/mengban.png')
mask=np.array(bg_fig)

for i in range(domain_num):
    
    stylecloud.gen_stylecloud(bg=mask,
                            text=text_domain[i],
                            # icon_name="fas fa-rectangle",
                            icon_name="fas fa-square",
                            background_color='white',
                            max_words=2000,
                            custom_stopwords=stopwords,
                            output_name=domain_dict[i]+'.png',
                            font_path="times.ttf",
                            gradient='horizontal')

    image = Image.open(domain_dict[i]+'.png')
    draw = ImageDraw.Draw(image)
    text = '('+chr(ord('a')+i)+') '+domain_dict_en[i]
    font = ImageFont.truetype('times.ttf', 30)
    position = (image.size[0]*0.4, 275)
    color = (0, 0, 0) 
    draw.text(position, text, font=font, fill=color)

    image_all.paste(image,box=(x,y))

    x+=512
    if x==3*512:
        x=0
        y+=316
    
    image.close()

image_all.save('CiYun_Weibo21.png')

