# BrickDL Library
***Deep learning inference library for graph-level optimizations on GPUs.***

## Organization

* `include`, `lib`, and `src` contains the BrickDL library source files.
* `benchmarks` has the implementations of all the baseline models
* `scripts` includes automated scripts to reproduce experiments 
* `evaluation` contains the generated figures (`.png`) and raw data (`.csv`) files from the experiments
* `cmake` CMake module file



## Software Dependencies

* CUDA 12.1.10
* PyTorch 1.12
* TensorFlow 2.14.1
* cuDNN v8.8.2
* Nsight Compute 2023.2.1
* Python ($>=$) 3.8
* CMake ($>=$) 3.17
* GCC ($>=$) 8.5

## Hardware Requirements
Any system consisting of a single NVIDIA A100 GPU with 40 GB HBM.

## Building the Library

1. Clone the repository. Create a build directory inside the source tree `mkdir build`
2. Create build configuration `cd build && cmake .. -DCMAKE_BUILD_TYPE=Release`



## Running the Experiments

### Experiment 1: Performance Evaluation (Section 4.2, Figure 7)
To evaluate BrickDL against the TensorFlow, PyTorch, cuDNN baselines for the seven models, run:
```
python scripts/model_perf.py
```
This generates `Fig7_modelperf.png` that corresponds to Figure 7 in the paper, and the raw data in `model_perf.csv`, both under the `evaluation/` directory.
This compares the execution time of BrickDL, TensorFlow, and PyTorch normalized to the cuDNN baseline for each model. It also contrasts the DRAM transaction time of BrickDL with cuDNN.

### Experiment 2: ResNet-50 Case Study (Section 4.4, Figures 8 & 9)
To compare the execution time of *padded bricks* and *memoized bricks* approaches of BrickDL with cuDNN for subgraph of ResNet-50: 
```
python scripts/resnet_eval.py
```
This generates two graphs: `Fig8_ResnetPerf.png`, plotted for the execution time of BrickDL approaches relative to cuDNN.
The second graph `Fig9_ResnetData.png` compares the data movement metrics for all versions.
The corresponding raw data `resnet_perf.csv` and `resnet_data.csv` can be found under `evaluation/`.

### Experiment 3: Analysis with Microbenchmarks (Section 4.5, Figures 10 & 11)
We evaluate BrickDL with two microbenchmarks.
The first microbenchmark evaluates BrickDL's approaches with a six-layer CNN layer partitioned into subgraphs of size 2, 3, 4, and 6. Run this script to compare BrickDL with cuDNN for all cases:
```
python scripts/Bench_GraphSize.py
```
This generates the graph `Fig10_GraphSize.png` that analyzes execution time, DRAM time, atomic operation time for all versions of graph partitioning.
The second microbenchmark compares BrickDL's approaches for different brick sizes for the effect of data padding and atomic operations. The following script generates `Fig11_BrickSize.png`:
```
python scripts/Bench_BrickSize.py
```










