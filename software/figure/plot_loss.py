
import os
import copy
import codecs
import random
import numpy as np
import pandas as pd
from optparse import OptionParser
import openpyxl
import sys
from matplotlib.pyplot import figure, show
import matplotlib.pyplot as plt
from matplotlib import cm, colors

import time
start = time.time()


################ m = 0
hls_float_m0_path="./pytorch-lr0.05-m0.0.txt"
hls_float_m0 = pd.read_csv(hls_float_m0_path , header=None).values

hls_float_m0 = np.array(hls_float_m0).ravel()
pytorch_m0 = hls_float_m0

hls_apfixed16_m0_path="./loss_save_b16i2_lr005_m0_apfixed"
hls_apfixed16_m0   = np.loadtxt(hls_apfixed16_m0_path)
hls_apfixed16_m0 = np.array(hls_apfixed16_m0).ravel()

hls_apfixed8_m0_path="./loss_save_b8i2_lr005_m0_apfixed"
hls_apfixed8_m0   = np.loadtxt(hls_apfixed8_m0_path)
hls_apfixed8_m0 = np.array(hls_apfixed8_m0).ravel()

################ m = 0.9
hls_float_m09_path="./pytorch-lr0.05-m0.9.txt"
hls_float_m09 = pd.read_csv(hls_float_m09_path , header=None).values
hls_float_m09 = np.array(hls_float_m09).ravel()

pytorch_m09 = hls_float_m09

hls_apfixed16_m09_path="./loss_save_b16i2_lr005_m09_apfixed"
hls_apfixed16_m09   = np.loadtxt(hls_apfixed16_m09_path)
hls_apfixed16_m09 = np.array(hls_apfixed16_m09).ravel()

hls_apfixed8_m09_path="./loss_save_b8i2_lr005_m09_apfixed"
hls_apfixed8_m09   = np.loadtxt(hls_apfixed8_m09_path)
hls_apfixed8_m09 = np.array(hls_apfixed8_m09).ravel()


fs_ = 12
plt.figure(3,figsize=(6,3))
length_min = len(hls_float_m0)
x_plot= range(length_min)

plt.scatter(x_plot, pytorch_m0, color='black', marker='.', alpha=1, s=10, label="Pytorch-float32", zorder=2)
plt.plot(x_plot, hls_apfixed16_m0, color = '#ff7f0e', label="Fpga-<16,2>", zorder=1)
plt.plot(x_plot, hls_apfixed8_m0, color = '#1f77b4', label="Fpga-<8,2>")

ymax = max(pytorch_m0)
ymin = min(pytorch_m0)
plt.legend(fontsize=fs_-1)
plt.margins(x=0,y=0)
plt.ylim(ymin-ymax*0.05,ymax)
plt.xlabel("Step",fontsize=fs_)
plt.ylabel("Trining Loss",fontsize=fs_)
plt.tick_params(labelsize=fs_)
plt.tight_layout()
plt.savefig("./multi_b_m0.pdf", format="pdf")
plt.show()




plt.figure(4,figsize=(6,3))
length_min = len(hls_float_m09)
x_plot= range(length_min)

plt.scatter(x_plot, pytorch_m09, color='black', marker='.', alpha=1, s=10, label="Pytorch-float32", zorder=2)
plt.plot(x_plot, hls_apfixed16_m09, color = '#ff7f0e', label="Fpga-<16,2>", zorder=1)
plt.plot(x_plot, hls_apfixed8_m09, color = '#1f77b4', label="Fpga-<8,2>")

ymax = max(pytorch_m09)
ymin = min(pytorch_m09)
plt.legend(fontsize=fs_-1)
plt.margins(x=0,y=0)
plt.ylim(ymin-ymax*0.05,ymax)
plt.xlabel("Step",fontsize=fs_)
plt.ylabel("Trining Loss",fontsize=fs_)
plt.tick_params(labelsize=fs_)
plt.tight_layout()
plt.savefig("./multi_b_m09.pdf", format="pdf")
plt.show()

