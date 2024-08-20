import matplotlib.pyplot as plt
import numpy as np
import csv



Total      = []
DRAM       = []
conflict   = []
compulsory = []
compute    = []
other      = []
idle       = []
labels     = ['cuDNN', 'B4 Padded', 'B4Memoized', 'B8 Padded',   'B8 Memoized',    'B16 Padded',   'B16 Memoized',  'B32 Padded', ' B32 Memoized']


x = np.arange(len(labels))
fig = plt.figure()
#ax = fig.add_axes([0,0,1,1])
fig, ax = plt.subplots(figsize=(9, 6))
width = 0.3

filename = open('evaluation/bench_bricksize.csv', 'r')
file = csv.DictReader(filename)
 
for col in file:
    DRAM.append(col['Memory'])
    compute.append(col['Compute'])
    conflict.append(col['Atomics (conflict)'])
    compulsory.append(col['Atomics (compulsory)'])
    idle.append(col['Idle'])
    other.append(col['Other'])

baseline = DRAM[0]+compute[0]
plt.axhline(y = baseline, color = 'deeppink', linestyle = 'dashed',  linewidth=0.85)
plt.vlines(x=0.5, color='darkgrey', linewidth=0.8, ymin=0, ymax=2200)
plt.vlines(x=2.5, color='darkgrey', linewidth=0.8, ymin=0, ymax=2200)
plt.vlines(x=4.5, color='darkgrey', linewidth=0.8, ymin=0, ymax=2200)
plt.vlines(x=6.5, color='darkgrey', linewidth=0.8, ymin=0, ymax=2200)

ax.set_axisbelow(True)
plt.grid(which='both', axis='y', zorder=0)
ax.grid(which = "minor", axis='y', linewidth = 0.1)
ax.minorticks_on()
ax.tick_params(which = "minor", bottom = False, left = False)


ax.bar(labels, DRAM, label='DRAM', color='salmon', edgecolor='black', linewidth=0.6, width=0.7)
ax.bar(labels, conflict, bottom=DRAM, label='Atomics (compulsory)', color='mediumaquamarine', edgecolor='olive', linewidth=0.6, hatch='\\\\\\', width=0.7)
ax.bar(labels, compulsory, bottom=np.add(DRAM,conflict), label='Atomics (conflict)', color='burlywood', edgecolor='chocolate', linewidth=0.6, hatch='///', width=0.7)
ax.bar(labels, compute, bottom=np.add(DRAM, np.add(conflict,compulsory)), label='Compute', color = 'cornflowerblue', edgecolor='blue', linewidth=0.6, hatch='...', width=0.7)
ax.bar(labels, other, bottom=np.add(np.add(DRAM,conflict),np.add(compulsory,compute)), label='Other', color='silver', edgecolor='gray', linewidth=0.6, hatch = 'xxx', width=0.7)


#ax.set_xlabel('Models')~
ax.set_ylabel("Execution time (ms)\n(Lower is Better)")

ax.set_xticks(x, labels = ["cuDNN\nBaseline", "$\mathregular{4^{3}}$ Brick\nPadded", "$\mathregular{4^{3}}$ Brick\nMemoized", "$\mathregular{8^{3}}$ Brick\nPadded", "$\mathregular{8^{3}}$ Brick\nMemoized", "$\mathregular{16^{3}}$ Brick\nPadded", "$\mathregular{16^{3}}$ Brick\nMemoized", "$\mathregular{32^{3}}$ Brick\nPadded", "$\mathregular{32^{3}}$ Brick\nMemoized"])
#ax.set_xticklabels(labels, fontsize=10, rotation=0)

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



ax.legend(loc=9, prop={'size': 9}, ncol=5)  

plt.title('Execution Time - 3-Layer Microbenchmark for Varying Brick Size', fontweight="bold", fontsize = 14)
#fig.tight_layout()
plt.rcParams['figure.figsize'] = [7, 9]


ax.set_ylim([0,1300])
plt.show()
fig.savefig('evaluation/Fig11_BrickSize.png',format='png', dpi=500, bbox_inches='tight')