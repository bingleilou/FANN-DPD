import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchsummary import summary
from pylfsr import LFSR
from bitstring import BitArray

from model_mlp import Full_Connect_MLP
import os
import random
import numpy as np
import pandas as pd
import sys
from matplotlib.pyplot import figure, show
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import time
from c_gen import *
from data_gen import batch_generator
start = time.time()


fpoly = [32,22,2,1]
state1 = [1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0]
state2 = [0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1]
L1 = LFSR(fpoly=fpoly,initstate =state1, verbose=False)
L2 = LFSR(fpoly=fpoly,initstate =state2, verbose=False)
# print the info
L1.info()
L2.info()


def  remove_windows(data, window_size):
        X = []
        win_size = int(window_size)
        for i in range(len(data)):
                win = data[i][0] + data[i][window_size]*1j
                X.append(win)
                i_index = i
        return np.array(X)
    
# set parameters
num_epochs = 1
batch_size = 32
log_every_n = 1
# Model
input_sizes_alone = 16
output_sizes = 2
hidden_size = 32
model = Full_Connect_MLP(input_sizes_alone,  output_sizes,  hidden_size)
    
# print parameters
print("model is: ",model)
print("len(list(model.parameters()))", len(list(model.parameters())))
for i in range(len(list(model.parameters()))):
    print(list(model.parameters())[i].size())

data_train_path = "./data/inout_OFDM_T40_1m_p4.txt"
data_test_path = data_train_path
data_train = pd.read_csv(data_train_path , header=None).values
data_test  = pd.read_csv(data_test_path , header=None).values

temp_test = np.genfromtxt(data_test_path, delimiter=',',dtype='str')
mapping_test = np.vectorize(lambda t:complex(t.replace('i','j')))
temp_train = np.genfromtxt(data_train_path, delimiter=',',dtype='str')
mapping_train = np.vectorize(lambda t:complex(t.replace('i','j')))

data_train= mapping_train(data_train)
data_test= mapping_test(data_test)


peak_in = 0
peak_out = 0

pa_in = data_train[:,0]
peak_in = max(max(abs(pa_in.real)),max(abs(pa_in.imag)))
pa_in_norm = pa_in/peak_in
pa_out = data_train[:,1]
peak_out = (max(max(abs(pa_out.real)),max(abs(pa_out.imag)))*1.1)
pa_out_norm = pa_out/peak_out
pa_in_test = data_test[:,0]
pa_in_test_norm = pa_in_test
pa_out_test = data_test[:,1]
pa_out_test_norm = pa_out_test
    
print('peak_in = ', peak_in);
print('peak_out = ', peak_out);

print("pa_in_norm.shape = ", pa_in_norm.shape)
print("pa_out_norm.shape = ",pa_out_norm.shape)
print("pa_in_test_norm.shape = ",pa_in_test_norm.shape)
window_size = int(input_sizes_alone/2)
window_size_y = int(output_sizes/2)


tr_len_temp = int(pa_in_norm.shape[0]*0.7)
tr_len = int(pa_in_norm.shape[0]*0.7)
tst_len = pa_in_norm.shape[0] - tr_len_temp
train_seq_append = []
train_seq_new_append = []
train_result_append = []
temp_append = []
test_seq_append = []
test_seq_new_append = []
train_inout_seq = np.zeros((int(pa_in_norm.shape[0])-window_size, input_sizes_alone+output_sizes))
test_in_seq = []
test_inout_seq = []


for i in range(tr_len+tst_len-window_size):
    train_seq    = pa_out_norm[i:i+window_size] # pa_in (t to t+w)
    train_result = pa_in_norm[i+window_size-window_size_y:i+window_size] # pa_out

    # list to  np.array
    train_seq    = np.array(train_seq)
    train_result = np.array(train_result)

    # seperate the real and imag 
    train_seq_i    = train_seq.real
    train_result_i = train_result.real
    train_seq_q    = train_seq.imag
    train_result_q = train_result.imag
    
    # reshape
    train_seq_i    = train_seq_i.reshape((-1,window_size))
    train_result_i = train_result_i.reshape((-1,window_size_y))
    train_seq_q    = train_seq_q.reshape((-1,window_size))
    train_result_q = train_result_q.reshape((-1,window_size_y))

    # no concatenate
    train_seq_new     = np.concatenate((train_seq_i, train_seq_q), axis=1)
    train_result_new  = np.concatenate((train_result_i, train_result_q), axis=1)

    # reshape
    train_seq_new = train_seq_new.reshape(window_size*2)
    train_result = train_result_new.reshape(window_size_y*2)

    # np.array to Tensor
    train_seq_new = torch.FloatTensor(train_seq_new)
    train_result = torch.FloatTensor(train_result)

    test_inout_seq.append((train_seq_new, train_result))
        
        
train_in_plot = []
train_out_plot = []
for i in range(int(pa_in_norm.shape[0])-window_size):
    train_seq    = pa_out_norm[i:i+window_size] # pa_in (t to t+w)
    train_result = pa_in_norm[i+window_size-window_size_y:i+window_size] # pa_out

    # list to  np.array
    train_seq    = np.array(train_seq)
    train_result = np.array(train_result)

    # seperate the real and imag 
    train_seq_i    = train_seq.real
    train_result_i = train_result.real
    train_seq_q    = train_seq.imag
    train_result_q = train_result.imag
    
    # no concatenate
    train_seq_new     = np.concatenate((train_seq_i, train_seq_q))
    train_result_new  = np.concatenate((train_result_i, train_result_q))

    # np.array to Tensor
    train_inout_seq[i] = np.concatenate((train_seq_new, train_result_new))
    
    # reshape
    train_seq_new_plot = train_seq_new.reshape(window_size*2)
    train_result_plot = train_result_new.reshape(window_size_y*2)

    train_in_plot.append(train_seq_new_plot)
    train_out_plot.append(train_result_plot)
    
train_in_plot = np.array(train_in_plot)
train_out_plot = np.array(train_out_plot)
test_inout_seq2  = test_inout_seq[tr_len_temp:tr_len_temp+tst_len]

print("train_inout_seq.shape = ",train_inout_seq.shape)

print("test_inout_seq2.len = ", len(test_inout_seq2))


print("train_in_plot[2,:] = ", train_in_plot[2,:])
print("train_out_plot[2,:] = ", train_out_plot[2,:])
print("  ")

# Loss & optimizer
criterion = nn.MSELoss(size_average=True)
optimizer = optim.SGD(model.parameters(), lr=0.05, momentum = 0.0)

print("  ")
# Train
model.train()
step_loss_train = []
epoch_loss_train = []

# Set the random seed manually for reproducibility.
seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

num_test = 102*batch_size*log_every_n
train_inout_seq2 = train_inout_seq[0:num_test]  #num_test  tr_len

min_loss_temp = 10000
n_count = 0

import copy
arr = copy.copy(train_inout_seq2)
all_sizes = input_sizes_alone+output_sizes
n_batches = int(len(arr) / batch_size)
arr = arr[:batch_size * n_batches]
arr_target = np.zeros((batch_size, all_sizes * n_batches))
for i in range(len(arr)):
    arr_target[i%batch_size, int(i/batch_size)*all_sizes:int(i/batch_size)*all_sizes+all_sizes] = arr[i]
print("  ")


g_train = batch_generator(train_inout_seq2, 1, input_sizes_alone, output_sizes)
train_in_batch = np.zeros((batch_size, input_sizes_alone))
train_out_batch = np.zeros((batch_size, output_sizes))
train_in_batch_store = np.zeros((batch_size, input_sizes_alone))
train_out_batch_store = np.zeros((batch_size, output_sizes))

train_in_batch_pool = []
train_out_batch_pool = []
size_pool = 10
for i in range(size_pool):
    train_in_batch_pool.append(np.zeros((batch_size, input_sizes_alone)))
    train_out_batch_pool.append(np.zeros((batch_size, output_sizes)))

min_loss_temp = 10000        
lfsr1 = 4042322160%size_pool
lfsr2 = 252645135%size_pool
L1.next()
#L2.next()
batch_flag = 0
for epoch in range(num_epochs):
    epoch_loss = 0
    n_count = 0
    index = 0
    
    batch_count_first = 0
    batch_count_shuffle = 0
    for [train_in, train_out] in g_train:
        index = index + 1
        
        if batch_flag == 0:
            # init BRAM 
            for i in range(size_pool):
                train_in_batch_pool[i][batch_count_first] = train_in
                train_out_batch_pool[i][batch_count_first] = train_out
            train_in_batch_store = copy.copy(train_in_batch_pool[lfsr2])
            train_out_batch_store = copy.copy(train_out_batch_pool[lfsr2])
        else:
            if index == 1:
                train_in_batch_store = copy.copy(train_in_batch_pool[lfsr2])
                train_out_batch_store = copy.copy(train_out_batch_pool[lfsr2])
                #print('********* index = ', index, 'coun = ', 'lfsr1 = ', lfsr1, 'lfsr2 = ', lfsr2, 'train_out_batch_store = \n', train_out_batch_store)
            else:
                train_in_batch_store = copy.copy(train_in_batch_store)
                train_out_batch_store = copy.copy(train_out_batch_store)
                
            # random store new samples to BRAM 
            train_in_batch_pool[lfsr1][0:batch_size-1] = train_in_batch_pool[lfsr1][1:batch_size]
            train_in_batch_pool[lfsr1][batch_size-1] = train_in
            train_out_batch_pool[lfsr1][0:batch_size-1] = train_out_batch_pool[lfsr1][1:batch_size]
            train_out_batch_pool[lfsr1][batch_size-1] = train_out
            seq1 = L1.state
            str1 = ''.join(map(str, seq1))
            b1 = BitArray(bin=str1)
            seq_int1 = b1.uint
            L1.next()
            lfsr1 = seq_int1%size_pool
            
            # random select new samples from BRAM 
            batch_count_shuffle = batch_count_shuffle + 1
            if batch_count_shuffle >= batch_size:
                batch_count_shuffle = 0
        
        batch_count_first = batch_count_first + 1
        if batch_count_first >= batch_size:
            batch_count_first = 0
            batch_flag = 1
            
            
        if index >= batch_size:
            index = 0
            seq2 = L2.state
            str2 = ''.join(map(str, seq2))
            b2 = BitArray(bin=str2)
            seq_int2 = b2.uint
            L2.next()
            lfsr2 = seq_int2%size_pool
            train_in_batch = copy.copy(train_in_batch_store)
            train_out_batch = copy.copy(train_out_batch_store)
            train_in_batch_tensor = torch.FloatTensor(train_in_batch)
            train_out_batch_tensor = torch.FloatTensor(train_out_batch)
            train_in_batch_tensor = train_in_batch_tensor.view(batch_size,  input_sizes_alone).requires_grad_()
            optimizer.zero_grad()
            outputs = model(train_in_batch_tensor)
            loss = criterion(outputs, train_out_batch_tensor)
            epoch_loss += loss.item()
            step_loss_train.append(loss.item()/batch_size)
            loss.backward()
            optimizer.step()

            if n_count%log_every_n == 0:
                print('Epoch %4d %4d / loss = %2.9f' % (epoch+1, n_count, loss.item()/batch_size))
                print('----------------------------------------------')
                epoch_loss_train.append(loss.item()/batch_size)
            n_count = n_count+1
            if loss.item()/batch_size < min_loss_temp:
                min_loss_temp  = loss.item()/batch_size
                min_loss_index  = n_count
 
if os.path.isdir('model/'):
    torch.save(model, os.path.join('model/model_%03d.pth' % num_epochs))


figure_1 = 1
if figure_1 == 1:
    plt.figure(1,figsize=(6,3))
    fs_ = 12
    length_min = len(epoch_loss_train)
    x_plot= range(length_min)
    plt.plot(x_plot, epoch_loss_train, color = 'dodgerblue')
    plt.xlabel("Step",fontsize=fs_)
    plt.ylabel("Training Loss",fontsize=fs_)
    plt.tick_params(labelsize=fs_)
    plt.tight_layout()
    loss_Df = pd.DataFrame(epoch_loss_train);
    loss_path_csv = "./figure/pytorch-lr0.05-m0.0.csv"
    loss_path_txt = "./figure/pytorch-lr0.05-m0.0.txt"
    loss_Df.to_csv(loss_path_csv);
    plt.show()
    
    ##################################################################################
    ## 1. Read original loss data file and generate the LIST: df_x
    ## 2. Read raw loss data file and generate the LIST: df_data
    ##################################################################################
    re_data = re.compile('^(.*)\,(.*)')
    df_data = []

    with open(loss_path_csv ,"r" ,encoding="utf-8") as f_r_dpd:
        data_split_as_lines = f_r_dpd .read().splitlines()

    for i in range(len(data_split_as_lines)):
        result = re_data.search(data_split_as_lines[i])
        if result:
            index = result.group(1)
            value = result.group(2)
            df_data.append([index, value])
    df_data = np.array(df_data)

    ##################################################################################
    ## 1. Delete the first line
    ## 2. Add several lines at the begaing of the data file
    ##################################################################################
    df_data = df_data.tolist()
    df_data.pop(0)
    df_data = np.array(df_data)
    df_data = df_data.tolist()
    df_data = np.array(df_data)

    ##################################################################################
    ## Write the loss data without line numbers to the "loss_path_txt"
    ##################################################################################
    with open(loss_path_txt, "w", encoding="utf-8") as f_w:
        for i in range(len(data_split_as_lines)+window_size-1):
            f_w.write(df_data[i,1])
            f_w.write('\n')
    
test = 1
if test == 1:
    model_test = Full_Connect_MLP(input_sizes_alone,  output_sizes,  hidden_size)
    model_test = torch.load('model/model_%03d.pth' % num_epochs)
    
    actual_in = []
    actual_out = []
    predicted_out = []
    
    model_test.eval()
    for test_in, test_out in test_inout_seq2:  
        preout = model_test(test_in)
        actual_in.append(test_in.detach().numpy())
        actual_out.append(test_out.detach().numpy())
        predicted_out.append(preout.detach().numpy())
    
    actual_in     = np.array(actual_in)
    actual_out    = np.array(actual_out)
    predicted_out = np.array(predicted_out)
    predicted_out = predicted_out.reshape(actual_out.shape)
    print("actual_in.shape = ", actual_in.shape)
    print("actual_out.shape = ",      actual_out.shape)
    print("predicted_out.shape = ", predicted_out.shape)

    
    print(" ")
    print("############## mat - remove_windows_ngspice ###########")
    actual_out_mat      = remove_windows(actual_out,      int(output_sizes/2))
    predicted_out_mat   = remove_windows(predicted_out,   int(output_sizes/2))
    print("actual_out_mat.shape = ", actual_out_mat.shape)
    print("predicted_out_mat.shape = ", predicted_out_mat.shape)
    print("actual_out_mat[0:5] = ", actual_out_mat[0:5])
    print("predicted_out_mat[0:5] = ", predicted_out_mat[0:5])
    NMSE_Test_Set = ((np.linalg.norm(actual_out_mat - predicted_out_mat))**2)/((np.linalg.norm(actual_out_mat))**2)
    NMSE_Test_Set_dB = 10*np.log10(NMSE_Test_Set);
    print('NMSE(Test Set): %f, NMSE dB(Test Set) %f' % (NMSE_Test_Set, NMSE_Test_Set_dB))
        




