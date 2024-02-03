import pandas as pd
import os
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

x=[i for i in range(1,11)]
fig=plt.figure(figsize=(24,8))

ax1,ax2=fig.add_subplot(121),fig.add_subplot(122)
fig.subplots_adjust(wspace=0.3)

path,color,label_name=('./Ch9cmdfendsample_times.xlsx','dodgerblue','sampling times')
file=pd.read_excel(path)
y= list(file.loc[:,'CM_avg_F1'])
l1=ax1.plot(x,y,c=color,marker='o',label=label_name,markeredgecolor='black',markersize=20)


path,color,label_name=('Ch9cmdfendheads_num.xlsx','dodgerblue','crossattention heads')
file=pd.read_excel(path)
y= list(file.loc[:,'CM_avg_F1'])
l2=ax2.plot(x,y,c=color,marker='o',label=label_name,markeredgecolor='black',markersize=20)

color,label_name=('darkorange','mdfend')
y= [0.9172 for i in range(1,11)]
l3=ax1.plot(x,y,c=color,marker='^',markeredgecolor='black',label=label_name,markersize=20, linestyle='--')
l4=ax2.plot(x,y,c=color,marker='^',markeredgecolor='black',label=label_name,markersize=20, linestyle='--')

ax1.legend(prop = {'family': 'Times New Roman','size':36},loc='lower right',frameon=True)
ax2.legend(prop = {'family': 'Times New Roman','size':36},loc='lower right',frameon=True)

ax1.set_xlim(0,11)
ax1.set_ylim(0.90,0.93)
ax1.set_xticks(ticks=range(1,11,1))
ax1.set_xticklabels([ str(i) for i in range(1,11,1) ],fontsize = 40)
ax1.set_yticks(ticks=[0.90+0.005*i for i in range(7)])
ax1.set_yticklabels([ str(0.90+0.005*i) for i in range(7) ],fontsize = 40) 
ax1.set_xlabel('(a) sample times T',fontdict={'family':'Times New Roman','size':48})
ax1.set_ylabel('F1',fontdict={'family':'Times New Roman','size':48})
ax1.grid(True,axis='y', linestyle='dotted',linewidth=4)


ax2.set_xlim(0,11)
ax2.set_ylim(0.90,0.93)
ax2.set_xticks(range(1,11,1))
ax2.set_xticklabels([ str(i) for i in range(1,11,1) ],fontsize = 40)
ax2.set_yticks([0.90+0.005*i for i in range(7)])
ax2.set_yticklabels([ str(0.90+0.005*i) for i in range(7) ],fontsize = 40) 
ax2.set_xlabel('(b) crossattention heads N',fontdict={'family':'Times New Roman','size':48})
ax2.set_ylabel('F1',fontdict={'family':'Times New Roman','size':48})
ax2.grid(True,axis='y', linestyle='dotted',linewidth=4)


plt.savefig('ch9'+'_hyperparametric_sensitivity_'+str(2016)+'.pdf', bbox_inches='tight')
    
    