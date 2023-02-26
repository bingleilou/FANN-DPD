# FANN-DPD
FPGA-based Adaptive Digital Predistorter for Power Amplifier with Neural Network Online-Training


--------------------------------------------------------------------------------
Introduction
--------------------------------------------------------------------------------
This is the first FPGA implementation of an adaptive DNN-DPD scheme for power amplifier linearization.
An FPGA implementation that supports 490 MHz streaming signal processing for DNN online training and inference in parallel, that achieves a throughput of 490 (complex) MS/s and latency of only 507 ns.

--------------------------------------------------------------------------------
Build
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


