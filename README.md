# GPU Performance Modeling with SimPy (H100 vs B200)

## Project Overview
This project implements a discrete-event simulation model using SimPy to compare the performance of NVIDIA H100 (Hopper) and B200 (Blackwell) GPUs for various AI workloads. The model incorporates key system components including storage (NVMe SSD), host memory, PCIe interface, and a detailed GPU architecture (Streaming Multiprocessors, Warp Schedulers, Warps, HBM).

The primary goal is to identify performance bottlenecks within the system (GPU computation, memory bandwidth, storage I/O, PCIe bandwidth) for different application scenarios and to quantify the performance uplift from H100 to B200.

## Modeled System Architecture
The simulation models the following components and their interactions:

-   **StorageSystem:** Simulates a high-speed NVMe SSD with configurable sequential/random bandwidth and latency.
-   **HostMemory:** Represents the CPU's main memory, acting as a buffer between storage and GPU.
-   **PCIeInterface:** Models the data transfer bandwidth between the host and the GPU.
-   **GPU:**
    -   **MemorySystem (HBM):** Simulates the GPU's High Bandwidth Memory with configurable capacity and bandwidth.
    -   **StreamingMultiprocessor (SM):** Contains Warp Schedulers and execution units. Models the maximum number of active warps.
    -   **WarpScheduler:** Manages the scheduling of Warps to execution units within an SM.
    -   **Warp:** The fundamental unit of execution (32 threads), which performs computation and memory access requests.

## GPU and Storage Parameters
The model uses the following parameters for H100, B200 (placeholder), and PCIe 5.0 NVMe SSDs:

### NVIDIA H100 GPU (Hopper Architecture)
-   **CUDA Cores (FP32):** 16,896
-   **Tensor Cores:** 4th Generation
-   **Memory:** HBM3, 80 GB, 3.35 TB/s Bandwidth
-   **L2 Cache:** 50 MB
-   **NVLink:** 900 GB/s
-   **PCIe:** Gen 5
-   **SMs:** 132, Max Warps/SM: 64

### NVIDIA B200 GPU (Blackwell Architecture - Placeholder)
-   **CUDA Cores (FP32):** 36,000
-   **Tensor Cores:** 5th Generation (assumed 2x H100 performance for relevant ops)
-   **Memory:** HBM3e, 192 GB, 8 TB/s Bandwidth
-   **L2 Cache:** 128 MB
-   **NVLink:** 1.8 TB/s
-   **PCIe:** Gen 5
-   **SMs:** 288, Max Warps/SM: 128

### PCIe 5.0 NVMe SSD
-   **Sequential Read:** 14.5 GB/s
-   **Sequential Write:** 14 GB/s
-   **Random Read IOPS (4KB):** 2.3 Million
-   **Random Write IOPS (4KB):** 1.6 Million
-   **Latency:** 10-100 microseconds

## Workload Scenarios
The simulation includes three representative AI workload scenarios:

1.  **Vector Database Search:** Simulates searching a large vector dataset, involving index loading, coarse-grained GPU search, candidate data loading, and fine-grained GPU computation. This workload highlights mixed I/O and compute demands.
2.  **LLM KV Cache Offloading:** Models the process of offloading large Key-Value caches to storage and swapping them back into GPU memory during Large Language Model (LLM) inference. This scenario is highly sensitive to storage latency and bandwidth.
3.  **GNN Training (Out-of-Core):** Simulates training Graph Neural Networks on datasets larger than GPU memory, requiring dynamic loading of graph partitions and features from storage. This is an I/O-bound workload.

## How to Run the Simulation

1.  **Prerequisites:** Ensure you have Python 3 and SimPy installed.
    ```bash
    pip install simpy
    ```

2.  **Execute the Simulation:**
    Navigate to the project directory and run the main simulation script:
    ```bash
    python3 run_simulation.py
    ```
    By default, the script runs simulations for both H100 and B200 GPUs with the `VectorDB_Search`, `LLM_KV_Cache_Offloading`, and `GNN_Training` workloads.

## Analyzing Results
The simulation output provides detailed logs for each step of the workload, including start time, end time, and duration. This allows for identifying bottlenecks (e.g., storage, PCIe, or GPU compute) and comparing the performance of H100 and B200 for each workload.

-   **Look for `Duration` values:** These indicate the time spent in each phase (e.g., `Index loading`, `GPU computation`).
-   **Compare H100 vs B200:** Observe how the total workload time and individual phase durations differ between the two GPU configurations.

## Future Work and Extensions
-   More detailed SM modeling (e.g., multiple execution units, register file, shared memory).
-   Modeling of NVLink for multi-GPU communication.
-   Implementation of more complex workload parameters and dynamic behaviors.
-   Visualization of simulation results.
-   Integration of different storage types (e.g., HDD, Optane).
