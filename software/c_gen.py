import numpy as np
import re
import string

def print_weights_biases_hls(data_path, model):
    re_dot = re.compile('^(.*).(.*)')
    for k, v in model.state_dict().items():
        #print('type(object) = ', type(k))
        np_v = np.array(v)
        np_v = np_v.transpose()
        shp = np_v.shape
        #ks=str(k).strip()
        maketrans = k.maketrans
        k = k.translate(maketrans('.', '_'))
        
        print('k = ', k, '; ', 'np_v.shape = ', shp, 'v.size = ', v.size())
        #print(np_v.ndim)
        f = open(data_path + str(k) + ".h", "w")
        if np_v.ndim == 1:
            f.write("static bias_t " + str(k) + "[" + str(shp[0]) + "] = {")
            for j in range(shp[0]-1):
                f.write(str(np_v[j]) + ",")
            f.write(str(np_v[shp[0]-1]) + "};\n")
        else:
            f.write("static weight_t " + str(k) + "[" + str(shp[0]) + "][" + str(shp[1]) + "] = {{")
            for i in range(shp[0]-1):
                for j in range(shp[1]-1):
                    f.write(str(np_v[i][j]) + ",")
                f.write(str(np_v[i][shp[1]-1]) + "},\n{")
            for j in range(shp[1]-1):
                f.write(str(np_v[shp[0]-1][j]) + ",")
            f.write(str(np_v[shp[0]-1][shp[1]-1]) + "}};\n")
            
        f.flush()
        f.close()
        
def print_weights_biases_matlab(data_path, model):
    re_dot = re.compile('^(.*).(.*)')
    for k, v in model.state_dict().items():
        #print('type(object) = ', type(k))
        np_v = np.array(v)
        shp = np_v.shape
        #ks=str(k).strip()
        maketrans = k.maketrans
        k = k.translate(maketrans('.', '_'))
        #print('k = ', k, '; ', 'np_v.shape = ', shp, 'v.size = ', v.size())
        #print(np_v.ndim)
        f = open(data_path + str(k) + "_matlab.txt", "w")
        if np_v.ndim == 1:
            for j in range(shp[0]-1):
                f.write(str(np_v[j]) + ",")
            f.write(str(np_v[shp[0]-1]) + "\n")
        else:
            for i in range(shp[0]-1):
                for j in range(shp[1]-1):
                    f.write(str(np_v[i][j]) + ",")
                f.write(str(np_v[i][shp[1]-1]) + "\n")
            for j in range(shp[1]-1):
                f.write(str(np_v[shp[0]-1][j]) + ",")
            f.write(str(np_v[shp[0]-1][shp[1]-1]) + "\n")
        f.flush()
        f.close()

def print_input_hls(data_path, input_array):
    f = open(data_path + "input.txt", "w")
    for i in range(input_array.shape[0]):
        for j in range(input_array.shape[1]-1):
            f.write(str(input_array[i][j]) + ",")
        f.write(str(input_array[i][input_array.shape[1]-1]) + "\n")
    f.flush()
    f.close()
    
def print_output_hls(data_path, input_array):
    f = open(data_path + "output.txt", "w")
    for i in range(input_array.shape[0]):
        for j in range(input_array.shape[1]-1):
            f.write(str(input_array[i][j]) + ",")
        f.write(str(input_array[i][input_array.shape[1]-1]) + "\n")
    f.flush()
    f.close()

