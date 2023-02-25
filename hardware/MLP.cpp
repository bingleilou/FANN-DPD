#include "MLP.h"

template<unsigned short DIM_X1, unsigned short DIM_Y1, unsigned short DIM_X2, unsigned short DIM_Y2, unsigned short DIM_X3, unsigned short DIM_Y3, unsigned short DIM_X4, unsigned short DIM_Y4>
void relu_FC_fwbw(
    param_t  inputs[DIM_X1],            //(i)
    param_t  labels[2],                 //(i)
    param_t  inputs_t[DIM_X1],          //(i)
    param_t  shortcuts[2],              //(i)
    param_t  shortcuts_t[2],            //(i)

    float    &loss,                     //(o)
    param_t  outputs_t[DIM_Y4]          //(o)
)
{
#pragma HLS ARRAY_PARTITION variable=inputs                  complete dim=1
#pragma HLS ARRAY_PARTITION variable=inputs_t                complete dim=1
#pragma HLS ARRAY_PARTITION variable=labels                  complete dim=1
#pragma HLS ARRAY_PARTITION variable=shortcuts               complete dim=1
#pragma HLS ARRAY_PARTITION variable=shortcuts_t             complete dim=1
#pragma HLS ARRAY_PARTITION variable=outputs_t               complete dim=1

#pragma HLS ARRAY_PARTITION variable=fc1_weight              complete dim=1
#pragma HLS ARRAY_PARTITION variable=fc2_weight              complete dim=1
#pragma HLS ARRAY_PARTITION variable=fc3_weight              complete dim=1
#pragma HLS ARRAY_PARTITION variable=fc4_weight              complete dim=1

    static param_t  outputs1[DIM_Y1] = {0};
    static param_t  outputs2[DIM_Y2] = {0};
    static param_t  outputs3[DIM_Y3] = {0};

    static param_t  activation1[DIM_X2] = {0};
    static param_t  activation2[DIM_X3] = {0};
    static param_t  activation3[DIM_X4] = {0};

    static param_t  activation1_t[DIM_X2] = {0};
    static param_t  activation2_t[DIM_X3] = {0};
    static param_t  activation3_t[DIM_X4] = {0};

    static bool relu_mask1[DIM_Y1] = {0};
    static bool relu_mask2[DIM_Y2] = {0};
    static bool relu_mask3[DIM_Y3] = {0};

    static bool relu_mask1_pipe[DIM_Y1] = {0};
    static bool relu_mask2_pipe[DIM_Y2] = {0};
    static bool relu_mask3_pipe[DIM_Y3] = {0};

#pragma HLS ARRAY_PARTITION variable=activation1             complete dim=1
#pragma HLS ARRAY_PARTITION variable=activation2             complete dim=1
#pragma HLS ARRAY_PARTITION variable=activation3             complete dim=1

#pragma HLS ARRAY_PARTITION variable=activation1_t           complete dim=1
#pragma HLS ARRAY_PARTITION variable=activation2_t           complete dim=1
#pragma HLS ARRAY_PARTITION variable=activation3_t           complete dim=1

#pragma HLS ARRAY_PARTITION variable=relu_mask1              complete dim=1
#pragma HLS ARRAY_PARTITION variable=relu_mask2              complete dim=1
#pragma HLS ARRAY_PARTITION variable=relu_mask3              complete dim=1

#pragma HLS ARRAY_PARTITION variable=outputs1                complete dim=1
#pragma HLS ARRAY_PARTITION variable=outputs2                complete dim=1
#pragma HLS ARRAY_PARTITION variable=outputs3                complete dim=1


    static weight_t velocities_w1[DIM_X1][DIM_Y1] = {0.0};
    static bias_t   velocities_b1[DIM_Y1] = {0.0};
    static accum_t delta_l_temp1[DIM_Y1] = {0.0};
    static accum_t delta_l_sum1[DIM_Y1] = {0.0};
    static accum_t activation_sum1[DIM_X1][DIM_Y1] = {0.0};

    static weight_t velocities_w2[DIM_X2][DIM_Y2] = {0.0};
    static bias_t   velocities_b2[DIM_Y2] = {0.0};
    static accum_t delta_l_temp2[DIM_Y2] = {0.0};
    static accum_t delta_l_sum2[DIM_Y2] = {0.0};
    static accum_t activation_sum2[DIM_X2][DIM_Y2] = {0.0};

    static weight_t velocities_w3[DIM_X3][DIM_Y3] = {0.0};
    static bias_t   velocities_b3[DIM_Y3] = {0.0};
    static accum_t delta_l_temp3[DIM_Y3] = {0.0};
    static accum_t delta_l_sum3[DIM_Y3] = {0.0};
    static accum_t activation_sum3[DIM_X3][DIM_Y3] = {0.0};

    static weight_t velocities_w4[DIM_X4][DIM_Y4] = {0.0};
    static bias_t   velocities_b4[DIM_Y4] = {0.0};
    static accum_t activation_sum4[DIM_X4][DIM_Y4] = {0.0};
    static accum_t delta_l_sum4[DIM_Y4] = {0.0};

#pragma HLS ARRAY_PARTITION variable=velocities_w1              complete dim=1
#pragma HLS ARRAY_PARTITION variable=delta_l_temp1              complete dim=1
#pragma HLS ARRAY_PARTITION variable=activation_sum1            complete dim=1

#pragma HLS ARRAY_PARTITION variable=velocities_w2              complete dim=1
#pragma HLS ARRAY_PARTITION variable=delta_l_temp2              complete dim=1
#pragma HLS ARRAY_PARTITION variable=activation_sum2            complete dim=1

#pragma HLS ARRAY_PARTITION variable=velocities_w3              complete dim=1
#pragma HLS ARRAY_PARTITION variable=delta_l_temp3              complete dim=1
#pragma HLS ARRAY_PARTITION variable=activation_sum3            complete dim=1

#pragma HLS ARRAY_PARTITION variable=velocities_w4              complete dim=1
#pragma HLS ARRAY_PARTITION variable=activation_sum4            complete dim=1

    static param_t velocities_w4_rd[DIM_X4][DIM_Y4] = {0};
    static param_t velocities_w3_rd[DIM_X3][DIM_Y3] = {0};
    static param_t velocities_w2_rd[DIM_X2][DIM_Y2] = {0};
    static param_t velocities_w1_rd[DIM_X1][DIM_Y1] = {0};

    static param_t activation_sum4_rd[DIM_X4][DIM_Y4] = {0};
    static param_t activation_sum3_rd[DIM_X3][DIM_Y3] = {0};
    static param_t activation_sum2_rd[DIM_X2][DIM_Y2] = {0};
    static param_t activation_sum1_rd[DIM_X1][DIM_Y1] = {0};

    static param_t velocities_b4_rd[DIM_Y4] = {0};
    static param_t velocities_b3_rd[DIM_Y3] = {0};
    static param_t velocities_b2_rd[DIM_Y2] = {0};
    static param_t velocities_b1_rd[DIM_Y1] = {0};

#pragma HLS ARRAY_PARTITION variable=velocities_w4_rd            complete dim=1
#pragma HLS ARRAY_PARTITION variable=velocities_w3_rd            complete dim=1
#pragma HLS ARRAY_PARTITION variable=velocities_w2_rd            complete dim=1
#pragma HLS ARRAY_PARTITION variable=velocities_w1_rd            complete dim=1

#pragma HLS ARRAY_PARTITION variable=activation_sum4_rd          complete dim=1
#pragma HLS ARRAY_PARTITION variable=activation_sum3_rd          complete dim=1
#pragma HLS ARRAY_PARTITION variable=activation_sum2_rd          complete dim=1
#pragma HLS ARRAY_PARTITION variable=activation_sum1_rd          complete dim=1

#pragma HLS ARRAY_PARTITION variable=velocities_b4_rd            complete dim=1
#pragma HLS ARRAY_PARTITION variable=velocities_b3_rd            complete dim=1
#pragma HLS ARRAY_PARTITION variable=velocities_b2_rd            complete dim=1
#pragma HLS ARRAY_PARTITION variable=velocities_b1_rd            complete dim=1

    static param_t  outputs[DIM_Y4] = {0};
#pragma HLS ARRAY_PARTITION variable=outputs            complete dim=1

    /////////////////////////////////////////
    //               layer-1               //
    /////////////////////////////////////////
    //Z = X*W + b
    //A = ReLU(Z) = ReLU(x*w+b)
    for (unsigned short coo = 0; coo < DIM_Y1; coo ++) {
#pragma HLS PIPELINE
        accum_t Z1 = 0;
        accum_t A1 = 0;
        accum_t Z1_t = 0;
        accum_t A1_t = 0;
        for (unsigned short cii = 0; cii < DIM_X1; cii++) {
#pragma HLS UNROLL
            Z1   += (accum_t)((accum_t)inputs[cii] * (accum_t)fc1_weight[cii][coo]);
            Z1_t += (accum_t)((accum_t)inputs_t[cii] * (accum_t)fc1_weight[cii][coo]);
        }
        Z1 += (accum_t)fc1_bias[coo];
        Z1_t += (accum_t)fc1_bias[coo];
        A1 = (Z1 > 0) ? Z1 : accum_t(0);
        A1_t = (Z1_t > 0) ? Z1_t : accum_t(0);
        relu_mask1[coo] = (Z1 > 0) ? bool(1) : bool(0);
        relu_mask1_pipe[coo] = (Z1 > 0) ? bool(1) : bool(0);
        activation1[coo] = (param_t)A1;
        activation1_t[coo] = (param_t)A1_t;
    }

    /////////////////////////////////////////
    //               layer-2               //
    /////////////////////////////////////////
    for (unsigned short coo = 0; coo < DIM_Y2; coo ++) {
#pragma HLS PIPELINE II=1
        accum_t Z2 = 0;
        accum_t A2 = 0;
        accum_t Z2_t = 0;
        accum_t A2_t = 0;
        for (unsigned short cii = 0; cii < DIM_X2; cii++) {
#pragma HLS UNROLL
            Z2 += (accum_t)((accum_t)activation1[cii] * (accum_t)fc2_weight[cii][coo]);
            Z2_t += (accum_t)((accum_t)activation1_t[cii] * (accum_t)fc2_weight[cii][coo]);
        }
        Z2 += (accum_t)fc2_bias[coo];
        Z2_t += (accum_t)fc2_bias[coo];
        A2 = (Z2 > 0) ? Z2 : accum_t(0);
        A2_t = (Z2_t > 0) ? Z2_t : accum_t(0);
        relu_mask2[coo] = (Z2 > 0) ? bool(1) : bool(0);
        relu_mask2_pipe[coo] = (Z2 > 0) ? bool(1) : bool(0);
        activation2[coo] = (param_t)A2;
        activation2_t[coo] = (param_t)A2_t;
    }

    /////////////////////////////////////////
    //               layer-3               //
    /////////////////////////////////////////
    for (unsigned short coo = 0; coo < DIM_Y3; coo ++) {
#pragma HLS PIPELINE II=1
        accum_t Z3 = 0;
        accum_t A3 = 0;
        accum_t Z3_t = 0;
        accum_t A3_t = 0;
        for (unsigned short cii = 0; cii <  DIM_X3; cii++) {
#pragma HLS UNROLL
            Z3 += (accum_t)((accum_t)activation2[cii] * (accum_t)fc3_weight[cii][coo]);
            Z3_t += (accum_t)((accum_t)activation2_t[cii] * (accum_t)fc3_weight[cii][coo]);
        }
        Z3 += (accum_t)fc3_bias[coo];
        Z3_t += (accum_t)fc3_bias[coo];
        A3 = (Z3 > 0) ? Z3 : accum_t(0);
        A3_t = (Z3_t > 0) ? Z3_t : accum_t(0);
        relu_mask3[coo] = (Z3 > 0) ? bool(1) : bool(0);
        relu_mask3_pipe[coo] = (Z3 > 0) ? bool(1) : bool(0);
        activation3[coo] = (param_t)A3;
        activation3_t[coo] = (param_t)A3_t;
    }

    /////////////////////////////////////////
    //               layer-4               //
    /////////////////////////////////////////
    for (unsigned short coo = 0; coo < DIM_Y4; coo ++) {
#pragma HLS PIPELINE II=1
        accum_t Z4 = 0;
        accum_t A4 = 0;
        accum_t Z4_t = 0;
        accum_t A4_t = 0;
        for (unsigned short cii = 0; cii < DIM_X4; cii++) {
#pragma HLS UNROLL
            Z4 += (accum_t)((accum_t)activation3[cii] * (accum_t)fc4_weight[cii][coo]);
            Z4_t += (accum_t)((accum_t)activation3_t[cii] * (accum_t)fc4_weight[cii][coo]);
        }
        Z4 += (accum_t)fc4_bias[coo];
        Z4_t += (accum_t)fc4_bias[coo];
        A4 = Z4;
        A4_t = Z4_t;
        outputs[coo] = (param_t)((param_t)A4 + shortcuts[coo]);
        outputs_t[coo] = (param_t)((param_t)A4_t + shortcuts_t[coo]);
    }


    /////////////////////////////////////////
    //                 MSE                 //
    /////////////////////////////////////////
    param_t error_[2] = {0};
    static unsigned short batch_count2 = 0;
    output_t loss_temp = 0;
    static output_t loss_sum = 0;

    output_t diff0 =  (output_t)labels[0] - (output_t)outputs[0];
    output_t diff1 =  (output_t)labels[1] - (output_t)outputs[1];

    loss_temp = (output_t)(diff0*diff0+diff1*diff1)*batch_size_div_2;
    error_[0] = (param_t)-diff0;
    error_[1] = (param_t)-diff1;

    batch_count2++;
    loss_sum += (output_t)loss_temp;

    if(batch_count2 >= BATCH_SIZE){
        loss = loss_sum;
        //printf("-------loss = %f\n",loss);
        loss_sum = 0;
        batch_count2 = 0;
    }

    /////////////////////////////////////////
    //                 back                //
    /////////////////////////////////////////
    static bool update_flag = 0;
    static unsigned short batch_count = 0;
    /////////////////////////////////////////
    //               layer-4               //
    /////////////////////////////////////////
    //calculate BP3
    CU4D_LOOP: for (unsigned short cii = 0; cii < DIM_Y4; cii ++) {
#pragma HLS PIPELINE II=1
        accum_t delta_l_sum4_read = delta_l_sum4[cii];
        accum_t error_rdiv = (accum_t)error_[cii]*batch_size_div;
        delta_l_sum4[cii] = delta_l_sum4_read + error_rdiv;
        for (unsigned short coo = 0; coo < DIM_X4; coo++) {
#pragma HLS UNROLL
            activation_sum4[coo][cii] +=  (accum_t)((accum_t)activation3[coo] * error_rdiv);
            delta_l_temp3[coo] +=  (!relu_mask3[coo]) ? (accum_t)0 : (accum_t)((accum_t)fc4_weight[coo][cii] * (accum_t)error_[cii]);
        }
    }

    /////////////////////////////////////////
    //               layer-3               //
    /////////////////////////////////////////
    //calculate BP3
    CU3D_LOOP: for (unsigned short cii = 0; cii < DIM_Y3; cii ++) {
#pragma HLS PIPELINE II=1
        accum_t delta_l_temp3_r = delta_l_temp3[cii];
        accum_t delta_l_temp3_rdiv = (accum_t)(delta_l_temp3[cii]*batch_size_div);
        delta_l_temp3[cii] = 0;
        delta_l_sum3[cii] += (!relu_mask3_pipe[cii]) ? (accum_t)0 : (accum_t)(delta_l_temp3_rdiv);
        for (unsigned short coo = 0; coo < DIM_X3; coo++) {
#pragma HLS UNROLL
            activation_sum3[coo][cii] += (!relu_mask3[cii]) ? (accum_t)0 : (accum_t)((accum_t)activation2[coo] * delta_l_temp3_rdiv);
            delta_l_temp2[coo] += (!relu_mask2[coo]) ? (accum_t)0 : (accum_t)((accum_t)fc3_weight[coo][cii] * delta_l_temp3_r);
        }
    }

    /////////////////////////////////////////
    //               layer-2               //
    /////////////////////////////////////////
    //calculate BP2
    CU2D_LOOP: for (unsigned short cii = 0; cii < DIM_Y2; cii ++) {
#pragma HLS PIPELINE II=1
        accum_t delta_l_temp2_r = delta_l_temp2[cii];
        accum_t delta_l_temp2_rdiv = (accum_t)(delta_l_temp2[cii]*batch_size_div);
        delta_l_temp2[cii] = 0;
        delta_l_sum2[cii] += (!relu_mask2_pipe[cii]) ? (accum_t)0 : (accum_t)(delta_l_temp2_rdiv);
        for (unsigned short coo = 0; coo < DIM_X2; coo++) {
#pragma HLS UNROLL
            activation_sum2[coo][cii] += (!relu_mask2[cii]) ? (accum_t)0 : (accum_t)((accum_t)activation1[coo] *delta_l_temp2_rdiv);
            delta_l_temp1[coo] += (!relu_mask1[coo]) ? (accum_t)0 : (accum_t)((accum_t)fc2_weight[coo][cii] * delta_l_temp2_r);
        }
    }

    /////////////////////////////////////////
    //               layer-1               //
    /////////////////////////////////////////
    //calculate BP1
    CU1D_LOOP: for (unsigned short cii = 0; cii < DIM_Y1; cii ++) {
#pragma HLS PIPELINE II=1
        accum_t delta_l_temp1_r = delta_l_temp1[cii];
        accum_t delta_l_temp1_rdiv = (accum_t)(delta_l_temp1[cii]*batch_size_div);
        delta_l_temp1[cii] = 0;
        delta_l_sum1[cii] += (!relu_mask1_pipe[cii]) ? (accum_t)0 : (accum_t)(delta_l_temp1_rdiv);
        for (unsigned short coo = 0; coo < DIM_X1; coo++) {
#pragma HLS UNROLL
            activation_sum1[coo][cii] += (!relu_mask1[cii]) ? (accum_t)0 : (accum_t)((accum_t)inputs[coo] * delta_l_temp1_rdiv);
        }
    }

    // mini-batch for SGD
    batch_count ++;
    if(batch_count >= BATCH_SIZE){
        batch_count = 0;
        update_flag = 1;
    }

    if(update_flag == 1){
        update_flag = 0;
        //update the weight by averaging the delta weight in the entire batch.
        loop_region0:{
            #pragma HLS LOOP_MERGE
            /////////////////////////////////////////
            //               layer-4               //
            /////////////////////////////////////////
            UP4_LOOP_0: for (unsigned short cii = 0; cii < DIM_Y4; cii ++) {
#pragma HLS PIPELINE II=1
                for (unsigned short coo = 0; coo < DIM_X4; coo++) {
                #pragma HLS UNROLL
                    velocities_w4_rd[coo][cii] = (param_t)MOM*velocities_w4[coo][cii];
                    activation_sum4_rd[coo][cii] = (param_t)activation_sum4[coo][cii];
                }
                velocities_b4_rd[cii] = (param_t)MOM*velocities_b4[cii];
            }

            UP4_LOOP: for (unsigned short cii = 0; cii < DIM_Y4; cii ++) {
#pragma HLS PIPELINE II=1
                for (unsigned short coo = 0; coo < DIM_X4; coo++) {
#pragma HLS UNROLL
                    velocities_w4[coo][cii] = velocities_w4_rd[coo][cii] + activation_sum4_rd[coo][cii];
                    weight_t velocities_w4_rd = (weight_t)LR * velocities_w4[coo][cii];
                    weight_t fc4_weight_rd = fc4_weight[coo][cii];
                    fc4_weight[coo][cii] = fc4_weight_rd + velocities_w4_rd;
                    activation_sum4[coo][cii] = 0.0;
                }
                velocities_b4[cii] = velocities_b4_rd[cii] + (param_t)(delta_l_sum4[cii]);

                bias_t fc4_bias_rd = fc4_bias[cii];
                fc4_bias[cii]  = fc4_bias_rd + (bias_t)LR * velocities_b4[cii];
                delta_l_sum4[cii] = 0.0;
            }

            /////////////////////////////////////////
            //               layer-3               //
            /////////////////////////////////////////
            UP3_LOOP_0: for (unsigned short cii = 0; cii < DIM_Y3; cii ++) {
#pragma HLS PIPELINE II=1
                for (unsigned short coo = 0; coo < DIM_X3; coo++) {
                #pragma HLS UNROLL
                    velocities_w3_rd[coo][cii] = (param_t)MOM*velocities_w3[coo][cii];
                    activation_sum3_rd[coo][cii] = (param_t)activation_sum3[coo][cii];
                }
                velocities_b3_rd[cii] = (param_t)MOM*velocities_b3[cii];
            }

            UP3_LOOP: for (unsigned short cii = 0; cii < DIM_Y3; cii ++) {
#pragma HLS PIPELINE II=1
                for (unsigned short coo = 0; coo < DIM_X3; coo++) {
#pragma HLS UNROLL
                    velocities_w3[coo][cii] = velocities_w3_rd[coo][cii] + activation_sum3_rd[coo][cii];
                    weight_t velocities_w3_rd = (weight_t)LR * velocities_w3[coo][cii];
                    weight_t fc3_weight_rd = fc3_weight[coo][cii];
                    fc3_weight[coo][cii] = fc3_weight_rd + velocities_w3_rd;
                    activation_sum3[coo][cii] = 0.0;
                }
                velocities_b3[cii] = velocities_b3_rd[cii] + (param_t)(delta_l_sum3[cii]);

                bias_t fc3_bias_rd = fc3_bias[cii];
                fc3_bias[cii]  = fc3_bias_rd + (bias_t)LR * velocities_b3[cii];
                delta_l_sum3[cii] = 0.0;
            }

            /////////////////////////////////////////
            //               layer-2               //
            /////////////////////////////////////////
            UP2_LOOP_0: for (unsigned short cii = 0; cii < DIM_Y2; cii ++) {
#pragma HLS PIPELINE II=1
                for (unsigned short coo = 0; coo < DIM_X2; coo++) {
                #pragma HLS UNROLL
                    velocities_w2_rd[coo][cii] = (param_t)MOM*velocities_w2[coo][cii];
                    activation_sum2_rd[coo][cii] = (param_t)activation_sum2[coo][cii];
                }
                velocities_b2_rd[cii] = (param_t)MOM*velocities_b2[cii];
            }

            UP2_LOOP: for (unsigned short cii = 0; cii < DIM_Y2; cii ++) {
#pragma HLS PIPELINE II=1
                for (unsigned short coo = 0; coo < DIM_X2; coo++) {
#pragma HLS UNROLL
                    velocities_w2[coo][cii] = velocities_w2_rd[coo][cii] + activation_sum2_rd[coo][cii];
                    weight_t velocities_w2_rd = (weight_t)LR * velocities_w2[coo][cii];
                    weight_t fc2_weight_rd = fc2_weight[coo][cii];
                    fc2_weight[coo][cii] = fc2_weight_rd + velocities_w2_rd;
                    activation_sum2[coo][cii] = 0.0;
                }
                velocities_b2[cii] = velocities_b2_rd[cii] + (param_t)(delta_l_sum2[cii]);

                bias_t fc2_bias_rd = fc2_bias[cii];
                fc2_bias[cii]  = fc2_bias_rd + (bias_t)LR * velocities_b2[cii];
                delta_l_sum2[cii] = 0.0;
            }

            /////////////////////////////////////////
            //               layer-1               //
            /////////////////////////////////////////
            UP1_LOOP_0: for (unsigned short cii = 0; cii < DIM_Y1; cii ++) {
#pragma HLS PIPELINE II=1
                for (unsigned short coo = 0; coo < DIM_X1; coo++) {
                #pragma HLS UNROLL
                    velocities_w1_rd[coo][cii] = (param_t)MOM*velocities_w1[coo][cii];
                    activation_sum1_rd[coo][cii] = (param_t)activation_sum1[coo][cii];
                }
                velocities_b1_rd[cii] = (param_t)MOM*velocities_b1[cii];
            }

            UP1_LOOP: for (unsigned short cii = 0; cii < DIM_Y1; cii ++) {
#pragma HLS PIPELINE II=1
                for (unsigned short coo = 0; coo < DIM_X1; coo++) {
#pragma HLS UNROLL
                    velocities_w1[coo][cii] = velocities_w1_rd[coo][cii] + activation_sum1_rd[coo][cii];
                    weight_t velocities_w1_rd = (weight_t)LR * velocities_w1[coo][cii];
                    weight_t fc1_weight_rd = fc1_weight[coo][cii];
                    fc1_weight[coo][cii] = fc1_weight_rd + velocities_w1_rd;
                    activation_sum1[coo][cii] = 0.0;
                }
                velocities_b1[cii] = velocities_b1_rd[cii] + (param_t)(delta_l_sum1[cii]);

                bias_t fc1_bias_rd = fc1_bias[cii];
                fc1_bias[cii]  = fc1_bias_rd + (bias_t)LR * velocities_b1[cii];
                delta_l_sum1[cii] = 0.0;
            }
        }
    }
}

template<unsigned int seed>
unsigned int pseudo_random() {
  static ap_uint<32> lfsr = seed;
  bool b_32 = lfsr.get_bit(32-32);
  bool b_22 = lfsr.get_bit(32-22);
  bool b_2 = lfsr.get_bit(32-2);
  bool b_1 = lfsr.get_bit(32-1);
  bool new_bit = b_32 ^ b_22 ^ b_2 ^ b_1;
  lfsr = lfsr >> 1;
  lfsr.set_bit(31, new_bit);

  return (lfsr.to_uint())%(unsigned int)POOL_SIZE;
}

void MLP(
        interface_t input[INPUT_SIZE],            // in
        interface_t label[OUTPUT_SIZE],           // in
        interface_t output[OUTPUT_SIZE],          // out
        float &loss                               // out
)
{
    #pragma HLS INTERFACE m_axi depth=32 port=input offset=slave bundle=IN
    #pragma HLS INTERFACE m_axi depth=32 port=label offset=slave bundle=LABEL
    #pragma HLS INTERFACE m_axi depth=32 port=output offset=slave bundle=OUT
    #pragma HLS INTERFACE m_axi depth=32 port=loss offset=slave bundle=LOSS
    #pragma HLS INTERFACE s_axilite port=return bundle=CTRL

    static param_t input_buf[INPUT_SIZE] = {0};
    static param_t label_buf[OUTPUT_SIZE] = {0};
#pragma HLS ARRAY_PARTITION variable=input_buf            complete dim=1
#pragma HLS ARRAY_PARTITION variable=label_buf            complete dim=1

    LOOP_Getinput: for (unsigned short coo = 0; coo < INPUT_SIZE; coo++) {
            #pragma HLS PIPELINE II=1
            input_buf[coo]= (param_t)input[coo];
        }
    LOOP_Getlabel: for (unsigned short coo = 0; coo < OUTPUT_SIZE; coo++) {
        #pragma HLS PIPELINE II=1
        label_buf[coo]= (param_t)label[coo];
    }

    static param_t input_buf_shuffle[INPUT_SIZE] = {0};
    static param_t label_buf_shuffle[OUTPUT_SIZE] = {0};
#pragma HLS ARRAY_PARTITION variable=input_buf_shuffle            complete dim=1
#pragma HLS ARRAY_PARTITION variable=label_buf_shuffle            complete dim=1

    static param_t input_buf_final[INPUT_SIZE] = {0};
    static param_t label_buf_final[OUTPUT_SIZE] = {0};
#pragma HLS ARRAY_PARTITION variable=input_buf_final            complete dim=1
#pragma HLS ARRAY_PARTITION variable=label_buf_final            complete dim=1

    static unsigned int lfsr1 = ((unsigned int)seed1)%((unsigned int)POOL_SIZE);
    static unsigned int lfsr2 = ((unsigned int)seed2)%((unsigned int)POOL_SIZE);

#pragma HLS ARRAY_PARTITION variable=BRAM_IN            complete dim=3
#pragma HLS ARRAY_PARTITION variable=BRAM_OUT           complete dim=3
#pragma HLS ARRAY_PARTITION variable=BRAM_IN_store      complete dim=2
#pragma HLS ARRAY_PARTITION variable=BRAM_OUT_store     complete dim=2
    static unsigned short batch_count_first = 0;
    static unsigned short batch_flag_first = 0;
    static unsigned short batch_count_shuffle = 0;

    if(batch_flag_first == 0){
        ////////////////////////////////////////////////////////////////////////////////////
        //////////                           init  BRAM                           //////////
        ////////////////////////////////////////////////////////////////////////////////////
        LOOP_Firstbatch_input: for (unsigned short pool_idx = 0; pool_idx < POOL_SIZE; pool_idx++) {
            #pragma HLS PIPELINE II=1
            for (unsigned short cii = 0; cii < INPUT_SIZE; cii++) {
                #pragma HLS UNROLL
                BRAM_IN[pool_idx][batch_count_first][cii] = input_buf[cii];
            }

            for (unsigned short cii = 0; cii < OUTPUT_SIZE; cii++) {
                #pragma HLS UNROLL
                BRAM_OUT[pool_idx][batch_count_first][cii] = label_buf[cii];
            }
        }
    }
    else{
        ////////////////////////////////////////////////////////////////////////////////////
        //////////                 random store new smaple to BRAM                //////////
        ////////////////////////////////////////////////////////////////////////////////////

        if(batch_count_first == 0){
            for (unsigned short batch_idx = 0; batch_idx < BATCH_SIZE; batch_idx++) {
                #pragma HLS PIPELINE II=1
                for (unsigned short coo = 0; coo < INPUT_SIZE; coo++) {
                    #pragma HLS UNROLL
                    BRAM_IN_store[batch_idx][coo] = BRAM_IN[lfsr2][batch_idx][coo];
                }

                for (unsigned short coo = 0; coo < OUTPUT_SIZE; coo++) {
                    #pragma HLS UNROLL
                    BRAM_OUT_store[batch_idx][coo] = BRAM_OUT[lfsr2][batch_idx][coo];
                }
            }
        }

        LOOP_Storeinput: for (unsigned short coo = 0; coo < BATCH_SIZE-1; coo++) {
            #pragma HLS PIPELINE II=1
            for (unsigned short cii = 0; cii < INPUT_SIZE; cii++) {
                #pragma HLS UNROLL
                BRAM_IN[lfsr1][coo][cii] = BRAM_IN[lfsr1][coo+1][cii];
            }
            for (unsigned short cii = 0; cii < OUTPUT_SIZE; cii++) {
                #pragma HLS UNROLL
                BRAM_OUT[lfsr1][coo][cii] = BRAM_OUT[lfsr1][coo+1][cii];
            }
        }
        for (unsigned short cii = 0; cii < INPUT_SIZE; cii++) {
            #pragma HLS UNROLL
            BRAM_IN[lfsr1][BATCH_SIZE-1][cii] = input_buf[cii];
        }

        for (unsigned short cii = 0; cii < OUTPUT_SIZE; cii++) {
            #pragma HLS UNROLL
            BRAM_OUT[lfsr1][BATCH_SIZE-1][cii] = label_buf[cii];

        }
        lfsr1 = pseudo_random<(unsigned int)seed1>();

        ////////////////////////////////////////////////////////////////////////////////////
        //////////                 random select new sample from BRAM                //////////
        ////////////////////////////////////////////////////////////////////////////////////
        LOOP_Selectinput: for (unsigned short coo = 0; coo < INPUT_SIZE; coo++) {
            #pragma HLS UNROLL
            input_buf_shuffle[coo]= BRAM_IN_store[batch_count_shuffle][coo];
        }
        LOOP_Selectlabel: for (unsigned short coo = 0; coo < OUTPUT_SIZE; coo++) {
            #pragma HLS UNROLL
            label_buf_shuffle[coo]= BRAM_OUT_store[batch_count_shuffle][coo];
        }
        batch_count_shuffle++;
        if(batch_count_shuffle >= BATCH_SIZE){
            batch_count_shuffle = 0;
            lfsr2 = pseudo_random<(unsigned int)seed2>();
        }
    }

    static param_t fc4_out[OUTPUT_SIZE] = {0};
    #pragma HLS ARRAY_PARTITION variable=fc4_out              complete dim=1

    /////////////////////////////////
    //////////// forward ////////////
    static param_t shortcuts[OUTPUT_SIZE]   = {0};
    static param_t shortcuts_t[OUTPUT_SIZE] = {0};
    #pragma HLS ARRAY_PARTITION variable=shortcuts              complete dim=1
    #pragma HLS ARRAY_PARTITION variable=shortcuts_t            complete dim=1

    float loss_ = 0;

    for (unsigned short coo = 0; coo < INPUT_SIZE; coo++) {
        #pragma HLS UNROLL
        input_buf_final[coo] = (batch_flag_first == 0) ? input_buf[coo] : input_buf_shuffle[coo];
    }
    for (unsigned short coo = 0; coo < OUTPUT_SIZE; coo++) {
        #pragma HLS UNROLL
        label_buf_final[coo] = (batch_flag_first == 0) ? label_buf[coo] : label_buf_shuffle[coo];
    }
    batch_count_first++;
    if(batch_count_first >= BATCH_SIZE){
        batch_count_first = 0;
        batch_flag_first = 1;
    }

    shortcuts[0] = (param_t)input_buf_final[(unsigned short)(INPUT_SIZE/2)-1];
    shortcuts[1] = (param_t)input_buf_final[(unsigned short)(INPUT_SIZE)-1];
    shortcuts_t[0] = (param_t)input_buf[(unsigned short)(INPUT_SIZE/2)-1];
    shortcuts_t[1] = (param_t)input_buf[(unsigned short)(INPUT_SIZE)-1];
    relu_FC_fwbw<INPUT_SIZE,32,32,64,64,32,32,OUTPUT_SIZE>(input_buf_final, label_buf_final, input_buf, shortcuts, shortcuts_t, loss_, fc4_out);

    loss = (float)loss_;
    LOOP_Output_2: for (unsigned short coo = 0; coo < OUTPUT_SIZE; coo++) {
        #pragma HLS UNROLL
        output[coo]= (interface_t)fc4_out[coo];
    }
}

