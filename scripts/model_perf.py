import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
import torch
import csv
import sys
import os
import subprocess as sp
 

cudnn = []
torch  = []
brickdl = []
xla = []
data_bdl = []
labels = []
compute_bdl = []
model_eval = ['resnet50', 'darknet53', 'DRN' '3Dresnet', 'VGG16', 'DeepCAM', 'Inception']
ncu_metrics = "sm__cycles_elapsed.avg, l1tex_t_bytes.sum, lts_t_bytes.sum,dram_bytes.sum, sm_sass_thread_inst_executed_op_fadd_pred_on.sum, sm_sass_thread_inst_executed_op_fmul_pred_on.sum, sm sass_thread_inst_executed_op_ffma_pred_on.sum"

for model in model_eval
    MODEL_DIR = tempfile.gettempdir()
    version = 1
    export_path = os.path.join(MODEL_DIR, str(version))
    sp.run ("torchserve", "--start --model-store /models --models all >> pt_eval.log 2>&1")
    os.environ["MODEL_DIR"] = MODEL_DIR
    sp.run ("nohup tensorflow_model_server",  "--rest_api_port=8501  --model_name=model --model_base_path='${MODEL_DIR}' >> tf_eval.log 2>&1")
    sp.run ("./benchmarks/cudnn_test", model)
    sp.run ("ncu",model,"--metrics=",ncu_metrics,"--target-processes all -o evaluation/ncu_prof.log")
    sp.run ("ncu",brick_model,"--metrics=",ncu_metrics,"--target-processes all -o evaluation/ncu_brick.log")

x = np.arange(len(labels))
fig = plt.figure()
#ax = fig.add_axes([0,0,1,1])
fig, ax = plt.subplots(figsize=(9, 6))
width = 0.3

plt.axhline(y = 1, color = 'deeppink', linestyle = '-.',  linewidth=0.75)

ax.set_axisbelow(True)
plt.grid(which='both', axis='y', zorder=0)
ax.grid(which = "minor", axis='y', linewidth = 0.05)
ax.minorticks_on()
ax.tick_params(which = "minor", bottom = False, left = False)

filename = open('evaluation/model_perf.csv', 'r')
file = csv.DictReader(filename)

for col in file:
    cudnn.append(col['cuDNN'])
    torch.append(col['Torchscript'])
    brickdl.append(col['BrickDL'])
    xla.append(col['TensorFlow'])
    data_bdl.append(col['Memory'])
    compute_bdl.append(col['Compute'])
 
rects2 = ax.bar(x - width, cudnn, color = '#DC143C', width=0.13, label='cuDNN', edgecolor='#8C000F')
rects3 = ax.bar(x - width/2, torch, color = 'm', width=0.13, hatch='...', label='TorchScript', edgecolor = 'black')
rects4 = ax.bar(x , xla, color = 'yellowgreen', width=0.13, hatch='///', label='TensorFlow XLA',edgecolor='darkolivegreen')
rects5 = ax.bar(x + width/2 , brickdl, color = 'cornflowerblue', width=0.13, hatch='\\\\\\', label='BrickDL',edgecolor='blue')



#ax.set_xlabel('Models')
ax.set_ylabel("Relative Performance\n(Higher is Better)")

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



legend = ax.legend(loc=9, prop={'size': 11}, ncol=3)  
plt.title('End-to-End Inference Performance - NVIDIA A100 GPU', fontweight="bold", fontsize = 14)
#fig.tight_layout()
plt.rcParams['figure.figsize'] = [7, 9]


ax.set_ylim([0,1.5])
plt.show()
fig.savefig('evaluation/fig7_modelperf',format='png', dpi=500, bbox_inches='tight')