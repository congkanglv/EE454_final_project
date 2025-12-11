# MLP (Multi-Layer Perceptron) Verilog Implementation — README

Project members: Congkang Lv, Cesar Pineda, Jessica Yao

## 1. Project Overview

This project implements a quantized two-layer Multi-Layer Perceptron (MLP) using Verilog/SystemVerilog.
The model architecture is:

Input dimension: 64 (int8)
Hidden layer: 8 units (Linear → ReLU), 32-bit activation
Output layer: 10 units (Linear), 32-bit logits
Final output: argmax over 10 classes

All weights and biases are exported from a quantized Python model and loaded into hardware using `$readmemh`.
The goal is to implement full forward inference (matrix multiply-accumulate, activation, and argmax) in Verilog and verify the design using ModelSim.

---

## 2. Project Structure

final_project/
│
├── mlp_top_q.sv                (Top-level MLP module, 64→8→ReLU→10)
├── linear_layer_q.sv           (Quantized linear layer: int8 weights + int32 accumulator)
├── relu.sv                     (ReLU activation, 32-bit)
├── argmax.sv                   (Argmax module for 10 classes)
├── tb_mlp_top_q.sv             (Testbench for simulation)
│
├── W1_q.mem                    (Layer 1 weights, 64×8 int8)
├── b1_q.mem                    (Layer 1 biases, 8 int32)
├── W2_q.mem                    (Layer 2 weights, 8×10 int8)
├── b2_q.mem                    (Layer 2 biases, 10 int32)
│
└── x_sample.mem                (One test input sample, 64-dim int8)

---

## 3. Explanation of Each File

### mlp_top_q.sv

Top-level inference module.
Runs Layer 1, applies ReLU, runs Layer 2, and computes the final prediction using argmax.
Includes an FSM to control sequential execution of both linear layers.

### linear_layer_q.sv

General quantized linear layer.
Uses int8 weights, int32 biases, and int32 accumulators.
Loads parameters from `.mem` files.
Outputs 32-bit results.

### relu.sv

32-bit ReLU activation.
Implements `dout = max(din, 0)`.

### argmax.sv

Outputs the index of the maximum value among the 10 logits.

### tb_mlp_top_q_simple.sv

Testbench for MLP inference.
Loads input sample, generates a start pulse, waits a fixed amount of time, and prints logits and predicted class.

---

## 4. Selecting and Quantizing Input in Python

The following Python code is used to extract a test sample, quantize it to int8, and generate the content for `x_sample.mem`:

```python
# Select one sample from the dataset
sample_x = X_test[0]    # float32 input

# Input quantization scale (use 1.0 if no scaling is applied)
SCALE_X = 1.0

# Convert to int8
sample_x_q = np.round(sample_x * SCALE_X).astype(np.int8)

print(sample_x_q.tolist())   # Copy this list into x_sample.mem
```

The contents of `x_sample.mem` should contain 64 lines, each storing one 8-bit value in hex, for example:

00
01
FF
...

This file will be loaded using:

```verilog
$readmemh("x_sample.mem", x_in);
```

---

## 5. Running Simulation (ModelSim)

Enter the project directory:

```
cd final_project/
```

Compile all modules:

```
vlog -sv mlp_top_q.sv linear_layer_q.sv relu.sv argmax.sv tb_mlp_top_q.sv
```

Launch ModelSim GUI:

```
vsim tb_mlp_top_q
```

Add waveform:

```
add wave -r sim:/tb_mlp_top_q/*
```

Run the simulation:

```
run 20us
```

---

## 6. Simulation Output

A typical simulation output is:

```
TB: Loading x_sample.mem ...
TB: Sampling outputs after 20us...
logits[0] = ...
...
logits[9] = ...
pred_class = 5
```

This indicates that for the input sample in `x_sample.mem`, the hardware MLP predicts class 5.
Waveforms show:

Layer 1 produces 32-bit hidden activations
ReLU zeros out negative values
Layer 2 generates 10 logits
`pred_class` stabilizes to 5

The hardware inference pipeline functions as expected.
![alt text](image.png)