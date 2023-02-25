#include <iostream>
#include <cstring>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "MLP.h"

using namespace std;

#define SMAPLE_NUM 60000

template <class dataType, unsigned int samples, unsigned int window>
int read_file(const char * filename, dataType  **data){
    FILE *fp;
    fp = fopen(filename, "r");
    if (fp == 0) {
        return -1;
    }
    // Read data from file
    float newval;
    for (int ii = 0; ii < samples; ii++){
        for (int jj = 0; jj < window-1; jj++) {
            if (fscanf(fp, "%f,", &newval) != 0){
                data[ii][jj] = (dataType) newval;
            } else {
                return -2;
            }
        }
        if (fscanf(fp, "%f\n", &newval) != 0){
            data[ii][window-1] = (dataType) newval;
        } else {
            return -2;
        }
    }
    fclose(fp);
    return 0;
}

template <class dataType, unsigned int samples>
int write_file_1D(const char * filename, dataType data[samples]) {
      FILE *fp;
      fp = fopen(filename, "w");
      if (fp == 0) {
        return -1;
      }

      // Write data to file
      for (int ii = 0; ii < samples; ii++){
        float newval = (float) data[ii];
        fprintf(fp, "%.16f\n", newval);
       }
       fclose(fp);
       return 0;
}


int main(int argc, char **argv)
{
    interface_t  **X;
    X = (interface_t **)malloc(sizeof(interface_t *) * SMAPLE_NUM);
    for (int i = 0; i<SMAPLE_NUM; i ++) {
        X[i] = (interface_t *)malloc(sizeof(interface_t) * INPUT_SIZE);
    }

    interface_t  **Y;
    Y = (interface_t **)malloc(sizeof(interface_t *) * SMAPLE_NUM);
    for (int i = 0; i<SMAPLE_NUM; i ++) {
        Y[i] = (interface_t *)malloc(sizeof(interface_t) * OUTPUT_SIZE);
    }

    float *loss_save;
    loss_save = (float *)malloc(sizeof(float) * (int)PLOT_NUM);

    int rval_x, rval_y, rval_loss = 0;
    rval_x = read_file<interface_t, SMAPLE_NUM, INPUT_SIZE>("data/input.txt", X);
    rval_y = read_file<interface_t, SMAPLE_NUM, OUTPUT_SIZE>("data/output.txt", Y);
    static int loss_save_count = 0;
    for (int k = 0; k < (int)PLOT_NUM*(int)BATCH_SIZE; k ++) {
        ////////////////////////////////
        //////// HARDWARE //////////////
        ////////////////////////////////
        interface_t input[INPUT_SIZE];
        interface_t label[OUTPUT_SIZE];
        interface_t output[OUTPUT_SIZE];

        float loss;
        interface_t error[OUTPUT_SIZE];

        for (int dim_x = 0; dim_x < (int)INPUT_SIZE; dim_x++) {
            input[dim_x] = X[k][dim_x];
        }
        for (int dim_y = 0; dim_y < (int)OUTPUT_SIZE; dim_y++) {
            label[dim_y] = Y[k][dim_y];
        }

        MLP(input, label, output, loss);

        static int batch_count_tb = 0;

        batch_count_tb++;
        if(batch_count_tb >= (int)BATCH_SIZE){
            batch_count_tb = 0;
            loss_save_count++;
            printf(" ================= iteration: %d || loss: (%.9f) =================",k+1, loss);
            printf("\n");
            loss_save[loss_save_count-1] = loss;
        }
    }
    loss_save_count = 0;
    //rval_loss = write_file_1D<float, (int)PLOT_NUM>("D:/DPD/mlp/data/loss_save", loss_save);

    return 0;
}
