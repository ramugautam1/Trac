
# Not fully functional, needs work
import pandas as pd
rd = pd.read_excel('/home/nirvan/Desktop/Projects/EcadMyo_08_all/Tracking_Result_EcadMyo_08/TrackingID2022-08-02 18:11:23.905684.xlsx',\
                   sheet_name='Sheet1',)
print(rd)

import math
indexlist = []
for ix in range(0,7497):
    for jx in range(0,80,2):
        if rd.iloc[ix,jx]==510 and ix not in indexlist:
            indexlist.append(ix)
            break
# starline()
for ind in (indexlist):
    # print(rd.iloc[ind,0], end= ', ') if not pd.isna(rd.iloc[ind,0]) else print(' ', end='')
    # print(rd.iloc[ind,0], end= ', ')
    for jj in range(0,80,2):
        if pd.isna(rd.iloc[ind,jj]):
            break
        print(rd.iloc[ind,jj], end=', ') if jj%2==0 else print('')
    print('')


import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
idl = [510,726,1642,2822,3254,3539]
arr = [[510, 510, 510, 510, 510, 510, 510, 510, 510, 510, 510, 510, 510],[510, 510, 726, 726, 726, 726, 726, 726, 726, 726, 726],[510, 726, 726, 726, 726, 726, 726, 1642, 1642, 1642, 1642, 1642, 1642, 1642, 1642,1642],[510, 510, 726, 726, 726, 726, 726, 1642, 1642, 1642, 1642, 1642, 1642, 1642, 1642, 2822, 2822, 2822,2822,2822],[510, 510, 726, 726, 726, 726, 726, 726, 1642, 1642, 1642, 1642, 1642, 1642, 1642, 1642, 2822, 2822, 2822, 3254, 3254, 3254, 3254, 3254, 3254, 3254, 3254,3254]]
# ,[510, 510, 726, 726, 726, 726, 726, 726, 1642, 1642, 1642, 1642, 1642, 1642, 1642, 1642, 2822, 2822, 2822, 3254, 3254, 3539, 3539,3539]
##################################################################################
fig=plt.figure(figsize=(17,4))
# ax=fig.add_subplot(111)
ax = plt.subplot()
ax.set_xlim(0,32)
ax.set_ylim(0,12)
ax.set_xlabel('Time Points (t)', fontsize=17, color='k')
mpl.rc('xtick', labelsize=17)
plt.xticks(rotation=90)
plt.yticks(color='w')
k=1
# colors=['r','g','b','c','m','y']
colors = ['#20e000','#123ec3','brown','gold','m','y']
for i in range(np.size(arr,axis=0)):
    l=0
    for j in range(np.size(arr[i])):
        if arr[i][j]==idl[i]:

            if j == np.size(arr[i])-1:
                ax.text(j+2,k, str(arr[i][j]),fontsize=17, rotation=90)
            else:
                plt.plot([j+2,j+1],[k+1,k+1],c='k',linewidth=2)

            ax.scatter(j+1,k+1, c=colors[i], s=400);
            if l==0 and i!=0:
                plt.plot([j+1,j],[k+1,k-1],c='r', linewidth=2)
                # ax.text(j+1,k-0.3,'split',c='r',fontsize=40)
                l+=1
    k+=2

plt.show()