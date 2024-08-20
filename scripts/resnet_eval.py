import matplotlib.pyplot as plt
import numpy as np
import csv

padded = []
atomic = []
labels= ['Subgraph 1',       'Subgraph 2',         'Subgraph 3',       'Subgraph 4',         'Subgraph 5',       'Subgraph 6',       'Subgraph 7',       'Subgraph 8',   'Subgraph 9']
sub_labels = ['Memory', 'Compute']

x = np.arange(len(labels))
fig = plt.figure()
#ax = fig.add_axes([0,0,1,1])
fig, ax = plt.subplots(figsize=(9, 5))
width = 0.4
filename = open('evaluation/resnet_perf.csv', 'r')
file = csv.DictReader(filename)

for col in file:
    padded.append(col['PaddedBricks'])
    atomic.append(col['MemoizedBricks'])

ax.set_axisbelow(True)
plt.grid(which='both', axis='y', zorder=0)
ax.grid(which = "minor", axis='y', linewidth = 0.1)
ax.minorticks_on()
ax.tick_params(which = "minor", bottom = False, left = False)


rects4 = ax.bar(x - 0.14 , padded, color = 'yellowgreen', width=0.3, hatch='///', edgecolor='darkolivegreen', label='Padded Bricks')
rects5 = ax.bar(x + 0.16, atomic, color = 'cornflowerblue', width=0.3, hatch='\\\\\\', edgecolor='blue', label='Recursive Memoized Bricks')



ax.set_xlabel('Subgraphs (SG) of ResNet-50')
ax.set_ylabel("% of Speedup over cuDNN Baseline\n(Higher is Better)")

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



ax.legend(loc=1, prop={'size': 11}, ncol=1)  
plt.title('Speedup over cuDNN Baseline - ResNet-50 Subgraphs', fontweight="bold", fontsize = 14)
#fig.tight_layout()
#plt.rcParams['figure.figsize'] = [7, 9]


ax.set_ylim([0,4.8])
plt.show()
fig.savefig('evaluation/Fig8_ResnetPerf.png',format='png', dpi=500, bbox_inches='tight')