import matplotlib.pyplot as plt
import numpy as np
import csv
 
glb  = []
L2   = []
dram = []
labels= ["cuDNN\nBaseline", "2+2+2\nPadded\nBricks", "2+2+2\nMemoized\nBricks", "3+3\nPadded\nBricks", "3+3\nMemoized\nBricks", "4+2\nPadded\nBricks", "4+2\nMemoized\nBricks", "6\nPadded\nBricks", "6\nMemoized\nBricks"]

x = np.arange(len(labels))
fig = plt.figure()
#ax = fig.add_axes([0,0,1,1])
fig, ax = plt.subplots(figsize=(9, 6))
width = 0.6
filename = open('evaluation/resnet_data.csv', 'r')
file = csv.DictReader(filename)
for col in file:
    glb.append(col['L1'])
    L2.append(col['L2'])
    dram.append(col['DRAM'])
    
ax.set_axisbelow(True)
plt.grid(which='both', axis='y', zorder=0)
ax.grid(which = "minor", axis='y', linewidth = 0.1)
ax.minorticks_on()
ax.tick_params(which = "minor", bottom = False, left = False)
 
rects4 = ax.bar(x - 0.2 , glb, color = 'mediumseagreen', width=0.2, hatch='///', edgecolor='darkolivegreen', label='Global (L1) Transactions')
rects5 = ax.bar(x , L2, color = 'mediumslateblue', width=0.2, hatch='\\\\\\', edgecolor='mediumblue', label='L2 Transactions')
rects6 = ax.bar(x + 0.2, dram, color = 'indianred', width=0.2, hatch='...', edgecolor='darkred', label='DRAM Transactions')

plt.vlines(x=0.5, color='darkgrey', linewidth=0.8, ymin=0, ymax=1600000)
plt.vlines(x=2.5, color='darkgrey', linewidth=0.8, ymin=0, ymax=1600000)
plt.vlines(x=4.5, color='darkgrey', linewidth=0.8, ymin=0, ymax=1600000)
plt.vlines(x=6.5, color='darkgrey', linewidth=0.8, ymin=0, ymax=1600000)


#ax.set_xlabel('Subgraphs (SG) of ResNet-50')
ax.set_ylabel("Number of Transactions\n(Lower is Better)")

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=10, rotation=0)

#plt.grid(which='both', axis='y', zorder=0)

#line1 = ax.plot(1, color='red', linestyle='dashed', linewidth=2, label='STREAM')

def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')



ax.legend(loc=9, prop={'size': 10}, ncol=3)  
plt.title('Data Movement - 6-Layer Microbenchmark', fontweight="bold", fontsize = 14)
#fig.tight_layout()
#plt.rcParams['figure.figsize'] = [7, 9]


ax.set_ylim([0,1600000])
plt.show()
fig.savefig('evaluation/Fig9_ResnetData.png',format='png', dpi=500, bbox_inches='tight')