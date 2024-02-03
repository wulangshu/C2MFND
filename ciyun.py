import pandas as pd
import os
import stylecloud
import jieba
import jieba.analyse
import palettable
import re
from PIL import Image, ImageDraw, ImageFont
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

text={i:'' for i in range(domain_num)}
words={i:[] for i in range(domain_num)}
for path in ['./data/train_cn.pkl','./data/val_cn.pkl','./data/test_cn.pkl']:
    file=pd.read_pickle(path)
    for i in range(domain_num):
        text_all=list(file.loc[file['category']== domain_dict[i],'content'])
        text[i]=''.join(text_all)

        text[i] = re.findall('[\u4e00-\u9fa5]+', text[i], re.S)
        text[i] = "/".join(text[i])
        text[i]=jieba.cut(str(text[i]),cut_all=False)
        for word in text[i]:
            if len(word)>1:
                words[i].append(word)

for i in range(domain_num):
    words[i]=' '.join(words[i])

image_all=Image.new("RGB", size=(512*3, 512*3))
x,y=0,0

for i in range(domain_num):
    
    stylecloud.gen_stylecloud(text=words[i],
                            icon_name="fas fa-square",
                            background_color='white',
                            max_words=2000,
                            output_name=domain_dict[i]+'.png',
                            font_path="msyh.ttc",
                            gradient='horizontal')

    image = Image.open(domain_dict[i]+'.png')
    draw = ImageDraw.Draw(image)
    text = chr(ord('a')+i)+') '+domain_dict_en[i]
    font = ImageFont.truetype('times.ttf', 30)
    position = (image.size[0]*0.4, 475)
    color = (0, 0, 0) 
    draw.text(position, text, font=font, fill=color)
    image_all.paste(image,box=(x,y))

    x+=512
    if x==3*512:
        x=0
        y+=512
    
    image_all.show()
    image.save(domain_dict[i]+'.png')
    image.close()

image_all.save('CiYun_Weibo21.png')

pass