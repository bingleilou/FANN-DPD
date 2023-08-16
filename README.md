# FANN-DPD
Adaptive Digital Predistorter for Power Amplifier with Neural Network Online Training on FPGAs


--------------------------------------------------------------------------------
Introduction
--------------------------------------------------------------------------------
* This is the first DPD scheme for power amplifier linearization based on a run-time trainable DNN. 

* A novel hardware-efficient batch shuffle scheme for online training of streaming data that achieves similar accuracy to standard offline mini-batch stochastic gradient descent (SGD).

* An FPGA implementation supports 340 MHz streaming signal processing for DNN-DPD, with a latency of 496 cycles for DNN online training and inference.

--------------------------------------------------------------------------------
Build (the code will be uploaded later..)
--------------------------------------------------------------------------------
**In Ubuntu Linux environment**

__(1).__ For HLS Project Implementation:
```
cd FANN-DPD/hardware/
make hls
```
__(2).__ For Python Execution:
```
cd FANN-DPD/software/
python3 train_and_test_mlp_SGD_pbacth_hls.py
```

--------------------------------------------------------------------------------
Results
--------------------------------------------------------------------------------
<p align="center">
  <img alt="Light" src="https://github.com/bingleilou/FANN-DPD/blob/main/software/figure/multi_b_m0.png" width="45%">
&nbsp; &nbsp; &nbsp; &nbsp;
  <img alt="Dark" src="https://github.com/bingleilou/FANN-DPD/blob/main/software/figure/multi_b_m09.png" width="45%">
</p>

