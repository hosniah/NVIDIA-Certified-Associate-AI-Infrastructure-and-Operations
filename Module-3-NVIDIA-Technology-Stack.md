# Module 3 — NVIDIA Technology Stack

## Learning Objectives

By the end of this module, you will be able to:

- Describe NVIDIA GPU architectures from Volta through Blackwell and their key innovations
- Differentiate between NVIDIA DGX, HGX, and MGX platforms
- Explain GPU scaling from single-node to SuperPOD deployments
- Describe the CUDA programming model and why it matters
- Identify core NVIDIA software libraries and their roles in the AI stack
- Explain NGC, NVIDIA AI Enterprise, NIM, and the NeMo framework
- Understand how the hardware and software stack work together end-to-end

---

## 3.1 NVIDIA GPU Architectures

Each generation of NVIDIA data center GPUs introduces architectural innovations that deliver step-function improvements in AI performance. Understanding these generations is essential for the NCA-AIIO exam.

### Architecture Timeline

```
 2017       2020         2022            2024
  │          │             │               │
  ▼          ▼             ▼               ▼
Volta  ──► Ampere ──►   Hopper    ──►  Blackwell
(V100)     (A100)     (H100/H200)     (B200/B100)
```

### Volta (V100) — 2017

The V100 was a watershed GPU that introduced **Tensor Cores**, purpose-built hardware units for the matrix multiply-accumulate operations at the heart of deep learning.

**Key Specifications:**
- 5,120 CUDA Cores, 640 Tensor Cores
- 32 GB HBM2 memory, 900 GB/s memory bandwidth
- 125 TFLOPS Tensor (FP16)
- NVLink 2.0 (300 GB/s)

**Why It Mattered:** Before Volta, deep learning ran on CUDA Cores — general-purpose parallel processors. Tensor Cores delivered up to 12x the throughput for deep learning operations, proving that specialized hardware could dramatically accelerate AI. This established the template: every subsequent NVIDIA architecture would feature more powerful Tensor Cores.

### Ampere (A100) — 2020

The A100 introduced two transformative features: **Multi-Instance GPU (MIG)** and **TF32 precision**.

**Key Specifications:**
- 6,912 CUDA Cores, 432 3rd-gen Tensor Cores
- 40 GB or 80 GB HBM2e, up to 2 TB/s memory bandwidth
- 312 TFLOPS Tensor (TF32), 624 TFLOPS (FP16)
- NVLink 3.0 (600 GB/s)
- PCIe Gen4

**Key Innovations:**

**Multi-Instance GPU (MIG):** Allows a single A100 to be partitioned into up to **7 isolated GPU instances**, each with its own compute, memory, and cache resources. MIG enables multi-tenant GPU sharing — different users or workloads can run on the same physical GPU without interference. This is critical for inference workloads and cloud environments where a full GPU is more than a single model needs.

**TF32 (TensorFloat-32):** A new numeric format that provides the range of FP32 with the speed of FP16. TF32 is the default precision on A100 Tensor Cores and delivers AI training speedups without requiring any code changes.

**Structural Sparsity:** The A100 can exploit fine-grained sparsity (structured patterns of zero values) in neural networks to achieve up to 2x additional speedup on Tensor Core operations.

### Hopper (H100) — 2022

The H100 introduced the **Transformer Engine**, purpose-built for the Transformer architecture that dominates modern AI.

**Key Specifications:**
- 16,896 CUDA Cores, 528 4th-gen Tensor Cores
- 80 GB HBM3, up to 3.35 TB/s memory bandwidth
- 989 TFLOPS (FP16 Tensor), 1,979 TFLOPS (FP8 Tensor with sparsity)
- NVLink 4.0 (900 GB/s)
- PCIe Gen5
- MIG support (up to 7 instances)

**Key Innovations:**

**Transformer Engine:** Automatically manages mixed precision during Transformer model training, dynamically switching between FP8 and FP16 on a per-layer, per-iteration basis. FP8 provides 2x the throughput of FP16 with minimal impact on model accuracy. The Transformer Engine handles the precision management transparently — users don't need to modify their training code.

**FP8 Precision:** A new 8-bit floating-point format (available in two variants: E4M3 for forward pass, E5M2 for gradients) that doubles Tensor Core throughput compared to FP16.

**DPX Instructions:** Hardware-accelerated dynamic programming instructions that speed up algorithms in genomics, graph analytics, and route optimization.

**H200 Variant:** The H200 (2024) uses the same Hopper architecture but upgrades to **141 GB of HBM3e memory** with 4.8 TB/s bandwidth — a memory-optimized variant designed for large language model inference where memory capacity is the bottleneck.

### Blackwell (B200) — 2024

The B200 represents the latest generation, built on the **Blackwell architecture** with a chiplet-based (multi-die) design.

**Key Specifications:**
- 2nd-gen Transformer Engine with FP4 support
- 192 GB HBM3e memory, up to 8 TB/s memory bandwidth
- Up to 20 PetaFLOPS (FP4) AI performance
- NVLink 5.0 (1.8 TB/s)
- 5th-gen Tensor Cores

**Key Innovations:**

**FP4 Precision:** A 4-bit floating-point format that doubles throughput compared to FP8. The 2nd-gen Transformer Engine manages FP4/FP8/FP16 precision dynamically during training and inference.

**Multi-Die Design:** The B200 uses a chiplet architecture — two GPU dies connected by a high-bandwidth chip-to-chip interconnect within a single package, effectively doubling the compute resources.

**NVLink 5.0:** At 1.8 TB/s bidirectional per GPU, NVLink 5 provides 2x the bandwidth of NVLink 4 in the H100, enabling faster multi-GPU scaling.

**Decompression Engine:** Hardware acceleration for compressed data decompression, reducing storage and memory bandwidth requirements for database and analytics workloads.

### Architecture Comparison

| Feature            | V100 (Volta) | A100 (Ampere)     | H100 (Hopper)           | B200 (Blackwell) |
| ------------------ | ------------ | ----------------- | ----------------------- | ---------------- |
| **Year**           | 2017         | 2020              | 2022                    | 2024             |
| **Tensor Cores**   | 1st Gen      | 3rd Gen           | 4th Gen                 | 5th Gen          |
| **Memory**         | 32 GB HBM2   | 80 GB HBM2e       | 80 GB HBM3              | 192 GB HBM3e     |
| **Memory BW**      | 900 GB/s     | 2 TB/s            | 3.35 TB/s               | 8 TB/s           |
| **NVLink BW**      | 300 GB/s     | 600 GB/s          | 900 GB/s                | 1,800 GB/s       |
| **Key Innovation** | Tensor Cores | MIG, TF32         | Transformer Engine, FP8 | FP4, Multi-Die   |
| **MIG Support**    | No           | Yes (7 instances) | Yes (7 instances)       | Yes              |

---

## 3.2 NVIDIA AI Systems

NVIDIA offers complete, purpose-built systems for AI workloads at various scales, from single servers to entire data center deployments.

### DGX Systems

**DGX** is NVIDIA's flagship **turnkey AI server** — a complete, pre-configured, and validated system designed for the most demanding AI workloads. DGX systems come fully integrated with hardware, software, and support.

**DGX H100:**
- 8× H100 SXM5 GPUs (640 GB total GPU memory)
- 2× Intel Xeon Scalable CPUs (or AMD EPYC)
- 4× NVSwitch chips providing all-to-all GPU connectivity
- NVLink 4.0: 900 GB/s per GPU
- 8× ConnectX-7 NICs (400 Gb/s InfiniBand or Ethernet each)
- 2× BlueField-3 DPUs
- Total system power: ~10.2 kW
- Software: DGX OS (Ubuntu-based), pre-installed NVIDIA AI software stack

**DGX B200:**
- 8× B200 GPUs (1.5 TB total GPU memory)
- Blackwell architecture with NVLink 5.0 (1.8 TB/s per GPU)
- Total NVLink bisection bandwidth: 14.4 TB/s
- Total system power: ~14.3 kW
- Direct liquid cooling required

### HGX Platform

**HGX** is NVIDIA's **GPU baseboard reference design** for OEM partners. Think of it as the "DGX engine" without NVIDIA's own chassis — Dell, HPE, Lenovo, and Supermicro build their own server platforms around HGX baseboards.

**HGX vs DGX:**

| Aspect            | DGX                                                       | HGX                                                  |
| ----------------- | --------------------------------------------------------- | ---------------------------------------------------- |
| **Sold by**       | NVIDIA directly                                           | OEM partners (Dell, HPE, Lenovo, etc.)               |
| **Form factor**   | Complete turnkey server                                   | GPU baseboard + OEM chassis                          |
| **Software**      | DGX OS + NVIDIA AI Enterprise                             | OEM's OS + NVIDIA drivers/stack                      |
| **Support**       | NVIDIA DGX support                                        | OEM support + NVIDIA driver support                  |
| **Customization** | Fixed configuration                                       | OEM can customize CPU, storage, networking           |
| **Use case**      | Organizations wanting a validated, ready-to-run AI system | Organizations with existing OEM vendor relationships |

### MGX Platform

**MGX (Modular GPU eXchange)** is NVIDIA's modular server reference architecture that supports flexible combinations of processors:
- Grace ARM CPU
- Any NVIDIA data center GPU
- BlueField DPU
- Mix of GPU counts (1, 2, 4, or 8)

MGX provides a standardized modular design that OEMs can use to build a wide range of server configurations from common building blocks, reducing time-to-market and design costs.

### Grace Hopper Superchip

The **Grace Hopper Superchip** combines NVIDIA's **Grace ARM CPU** and **Hopper GPU** on a single module, connected via NVLink-C2C at 900 GB/s with a unified memory architecture. This eliminates the PCIe bottleneck between CPU and GPU and enables the CPU and GPU to access each other's memory transparently.

The Grace CPU uses ARM Neoverse V2 cores with LPDDR5X memory (up to 480 GB), providing excellent energy efficiency compared to x86 CPUs.

### Scaling: POD and SuperPOD

NVIDIA provides reference architectures for scaling GPU systems from individual servers to data center-scale deployments:

**DGX POD:**
A cluster of approximately **20 DGX systems** connected via InfiniBand networking with shared storage. A DGX H100 POD delivers approximately 160 H100 GPUs — sufficient for training most enterprise AI models.

**DGX SuperPOD:**
A larger-scale deployment of **32 or more DGX PODs** (256+ DGX systems, 2,000+ GPUs) connected via a high-bandwidth InfiniBand fabric. SuperPODs are designed for the largest AI training workloads — training frontier language models, climate simulations, and scientific research.

```
Scale-Up Path:
Single GPU → Multi-GPU Server → DGX/HGX → DGX POD → DGX SuperPOD
(1 GPU)     (2-8 GPUs)        (8 GPUs)   (~160 GPUs) (2,000+ GPUs)
```

---

## 3.3 The CUDA Programming Model

**CUDA (Compute Unified Device Architecture)** is NVIDIA's parallel computing platform and programming model. Released in 2006, CUDA is what transformed NVIDIA GPUs from graphics-only processors into general-purpose accelerators — and it is the foundational layer that the entire NVIDIA AI software stack is built upon.

### Why CUDA Matters

Before CUDA, using a GPU for non-graphics computation required encoding your problem as a graphics operation (shader programs) — impractical for most scientific and AI workloads. CUDA allows developers to write standard C/C++/Python code with extensions that specify which portions should run on the GPU.

### CUDA Execution Model

**Key Concepts:**
- **Host**: The CPU and its memory (system RAM)
- **Device**: The GPU and its memory (HBM)
- **Kernel**: A function that runs on the GPU. A single kernel launch creates thousands or millions of parallel threads
- **Thread**: The smallest unit of execution on the GPU
- **Block**: A group of threads that can cooperate and share fast memory (shared memory)
- **Grid**: A collection of blocks that execute the same kernel

**Execution Flow:**
1. CPU prepares data and transfers it to GPU memory
2. CPU launches a kernel on the GPU
3. GPU executes the kernel across thousands of threads in parallel
4. Results are transferred back to CPU memory (or used directly on the GPU for the next operation)

### CUDA Ecosystem

CUDA is not just a programming model — it is an entire ecosystem of libraries, tools, and frameworks:
- **CUDA Toolkit**: Compiler (nvcc), debugger (cuda-gdb), profiler (Nsight), and runtime libraries
- **cuDNN, cuBLAS, cuFFT**: Optimized libraries built on CUDA (see Section 3.4)
- **Framework Integration**: PyTorch, TensorFlow, and JAX all use CUDA under the hood for GPU acceleration

The vast majority of AI practitioners never write CUDA code directly — they use frameworks like PyTorch that call CUDA-optimized libraries. But CUDA is always running underneath.

---

## 3.4 NVIDIA Software Stack for AI

NVIDIA's software ecosystem is a layered stack, from low-level GPU libraries to high-level application frameworks. Each layer builds on the one below it.

```
┌─────────────────────────────────────────────────────────┐
│                Applications & Frameworks                │
│       (NeMo, Metropolis, Clara, DRIVE, Omniverse)       │
├─────────────────────────────────────────────────────────┤
│                  Deployment & Serving                   │
│        (Triton Inference Server, NIM, TensorRT)         │
├─────────────────────────────────────────────────────────┤
│                      AI Frameworks                      │
│           (PyTorch, TensorFlow, JAX, RAPIDS)            │
├─────────────────────────────────────────────────────────┤
│                   Core GPU Libraries                    │
│          (cuDNN, cuBLAS, cuFFT, NCCL, CUTLASS)          │
├─────────────────────────────────────────────────────────┤
│                      CUDA Platform                      │
│              (CUDA Toolkit, Drivers, NVML)              │
├─────────────────────────────────────────────────────────┤
│                      GPU Hardware                       │
│         (CUDA Cores, Tensor Cores, HBM, NVLink)         │
└─────────────────────────────────────────────────────────┘
```

### Core GPU Libraries

These are highly optimized libraries that provide the fundamental building blocks for AI computation:

**cuDNN (CUDA Deep Neural Network Library)**
Provides optimized implementations of standard deep learning operations: convolutions, pooling, normalization, activation functions, and recurrent neural network primitives. PyTorch and TensorFlow use cuDNN for their GPU-accelerated operations. When you train a model in PyTorch, cuDNN is doing the heavy lifting.

**cuBLAS (CUDA Basic Linear Algebra Subroutines)**
The GPU-accelerated version of BLAS (Basic Linear Algebra Subroutines). Provides optimized matrix multiply, vector operations, and other linear algebra routines. Matrix multiplication — the core operation in Transformer attention — runs through cuBLAS.

**cuFFT (CUDA Fast Fourier Transform)**
GPU-accelerated FFT library for signal processing, audio analysis, and scientific computing.

**NCCL (NVIDIA Collective Communications Library)**
The multi-GPU communication library. NCCL provides optimized implementations of collective operations — **all-reduce, all-gather, broadcast, reduce-scatter** — that are essential for distributed AI training.

When training a model across multiple GPUs, each GPU computes gradients on its portion of the data. These gradients must be aggregated (all-reduce) across all GPUs before updating the model weights. NCCL handles this communication, automatically selecting the optimal algorithm and routing based on the GPU topology (NVLink, NVSwitch, PCIe, InfiniBand).

**NCCL is topology-aware**: it understands the physical interconnect topology (which GPUs are connected via NVLink vs. PCIe, which nodes are on the same InfiniBand switch) and selects the most efficient communication pattern.

**CUTLASS (CUDA Templates for Linear Algebra Subroutines)**
A collection of C++ template abstractions for implementing high-performance matrix multiply and convolution operations on NVIDIA GPUs. Used by developers building custom GPU kernels.

### Inference Optimization

**TensorRT**
NVIDIA's high-performance deep learning inference optimization SDK. TensorRT takes a trained model and optimizes it for deployment:
- **Layer and tensor fusion**: Combines multiple sequential operations into single GPU kernels
- **Precision calibration**: Automatically converts models from FP32 to FP16, INT8, or FP8 with minimal accuracy loss
- **Kernel auto-tuning**: Selects the optimal GPU kernel for each operation based on the specific GPU architecture
- **Dynamic tensor memory**: Optimizes GPU memory allocation during inference

TensorRT can deliver 2–5x inference speedup over running the original framework model directly.

**Triton Inference Server**
An open-source, production-grade inference serving software that simplifies deploying AI models at scale:
- **Multi-framework support**: Serves models from PyTorch, TensorFlow, TensorRT, ONNX Runtime, and custom backends
- **Dynamic batching**: Automatically groups multiple inference requests into batches for optimal GPU utilization
- **Model ensemble**: Chains multiple models together in a pipeline (preprocessing → model → postprocessing)
- **Concurrent model execution**: Runs multiple models on the same GPU simultaneously
- **Model versioning**: Supports multiple versions of a model, enabling A/B testing and gradual rollout
- **Metrics and monitoring**: Built-in Prometheus metrics for latency, throughput, and GPU utilization

**NVIDIA NIM (NVIDIA Inference Microservice)**
NIM packages a specific foundation model (LLM) as an optimized, ready-to-deploy container:
- Built on top of Triton Inference Server and TensorRT-LLM
- Provides an OpenAI-compatible API for easy integration
- Pre-optimized for specific GPU architectures
- Available for popular models (Llama, Mistral, etc.)

NIM simplifies deployment: instead of manually optimizing a model with TensorRT and configuring Triton, you pull a NIM container and it's ready to serve.

### Data Science

**RAPIDS**
A suite of open-source GPU-accelerated data science libraries:
- **cuDF**: GPU-accelerated DataFrames (pandas equivalent)
- **cuML**: GPU-accelerated machine learning algorithms (scikit-learn equivalent)
- **cuGraph**: GPU-accelerated graph analytics
- **cuSpatial**: GPU-accelerated spatial and geospatial analysis

RAPIDS accelerates the data preparation and feature engineering stages of the AI pipeline, which often consume more time than model training itself.

**DALI (Data Loading Library)**
A GPU-accelerated data loading and augmentation library. DALI offloads image decoding, resizing, cropping, and augmentation from the CPU to the GPU, keeping the data pipeline fast enough to saturate GPU compute during training.

---

## 3.5 Platforms and Ecosystem

### NGC Catalog

**NGC (NVIDIA GPU Cloud)** is NVIDIA's hub for GPU-optimized software. Despite the name containing "Cloud," NGC is a software catalog — not a cloud provider. It provides:

- **Containers**: Pre-built, optimized Docker containers for AI frameworks (PyTorch, TensorFlow), inference engines (Triton), and applications — tested and validated on NVIDIA GPUs
- **Pre-trained Models**: Ready-to-use AI models for common tasks (object detection, NLP, speech recognition)
- **Helm Charts**: Kubernetes deployment configurations for NVIDIA software
- **Model Scripts**: Training scripts and recipes for reproducing model results
- **SDKs and Tools**: Development tools, profilers, and utilities

NGC containers are the recommended way to deploy NVIDIA AI software — they ensure correct driver versions, library compatibility, and optimal performance.

### NVIDIA AI Enterprise

**NVIDIA AI Enterprise (NVAIE)** is NVIDIA's enterprise software platform that provides:
- **Certified, supported containers**: Enterprise-grade versions of NVIDIA AI software with SLAs and long-term support
- **vGPU Software**: GPU virtualization for running multiple AI workloads on shared GPU infrastructure
- **Security patches and updates**: Regular security and bug fix releases
- **Enterprise support**: Direct NVIDIA technical support

NVAIE is the enterprise packaging of NVIDIA's AI software stack — the same technology as the free NGC containers, but with enterprise support, certifications, and virtualization capabilities.

### Base Command Manager (BCM)

**Base Command Manager** is NVIDIA's infrastructure management software for AI clusters. It provides:
- **Cluster provisioning**: Automated deployment and configuration of GPU servers
- **Workload management**: Integration with Kubernetes and Slurm for job scheduling
- **Monitoring**: Cluster-wide monitoring of GPU health, utilization, and performance
- **User management**: Multi-tenant access control and resource allocation
- **Software management**: Centralized deployment and updates of drivers, CUDA toolkit, and frameworks

BCM is used to manage DGX POD and SuperPOD deployments.

### NeMo Framework

**NVIDIA NeMo** is an end-to-end framework for building, customizing, and deploying large language models (LLMs) and other generative AI models:
- **NeMo Curator**: Data curation and cleaning for LLM training datasets
- **NeMo Trainer**: Distributed training with support for tensor parallelism, pipeline parallelism, and data parallelism
- **NeMo Customizer**: Fine-tuning, PEFT (Parameter-Efficient Fine-Tuning), and RLHF (Reinforcement Learning from Human Feedback)
- **NeMo Evaluator**: Model evaluation and benchmarking
- **NeMo Guardrails**: Safety and alignment tools for LLM deployment

### Run:ai

**Run:ai** (acquired by NVIDIA) provides **GPU orchestration and scheduling** for Kubernetes-based AI infrastructure:
- Fractional GPU allocation (share a single GPU across multiple workloads)
- GPU pooling across clusters
- Fair-share scheduling with priorities and quotas
- GPU utilization optimization

### NVIDIA Omniverse

**Omniverse** is NVIDIA's platform for building and operating digital twins and 3D simulation environments:
- Used in manufacturing, robotics, and autonomous vehicle development
- Physics-accurate simulation powered by GPUs
- Based on Universal Scene Description (OpenUSD)
- Enables collaboration on 3D workflows

---

## 3.6 Industry-Specific Platforms

NVIDIA provides vertical-specific platforms built on top of the core technology stack:

| Platform              | Industry            | Purpose                                                      |
| --------------------- | ------------------- | ------------------------------------------------------------ |
| **NVIDIA DRIVE**      | Automotive          | Autonomous vehicle development (hardware + software)         |
| **NVIDIA Clara**      | Healthcare          | Medical imaging, genomics (Clara Parabricks), drug discovery |
| **NVIDIA Metropolis** | Smart Cities/Retail | Video analytics and intelligent video applications           |
| **NVIDIA Isaac**      | Robotics            | Robot simulation, perception, and manipulation               |
| **NVIDIA cuOpt**      | Logistics           | Route optimization and fleet management                      |

---

## Module 3 Summary

| Concept                  | Key Takeaway                                                                                                |
| ------------------------ | ----------------------------------------------------------------------------------------------------------- |
| **GPU Generations**      | Volta (Tensor Cores) → Ampere (MIG, TF32) → Hopper (Transformer Engine, FP8) → Blackwell (FP4, multi-die)   |
| **DGX**                  | Turnkey NVIDIA AI server (8 GPUs, NVLink, NVSwitch, DPU). DGX H100: ~10.2 kW. DGX B200: ~14.3 kW.           |
| **HGX vs DGX**           | HGX = GPU baseboard for OEMs. DGX = complete NVIDIA-branded server. Same GPU platform, different packaging. |
| **Scaling**              | GPU → Server → DGX POD (~20 nodes) → SuperPOD (32+ PODs, 2,000+ GPUs)                                       |
| **CUDA**                 | NVIDIA's parallel computing platform. Foundation of the entire AI software stack.                           |
| **cuDNN/cuBLAS**         | Optimized DNN and linear algebra libraries. PyTorch/TensorFlow use these under the hood.                    |
| **NCCL**                 | Multi-GPU communication library. Topology-aware all-reduce for distributed training.                        |
| **TensorRT**             | Inference optimization SDK. Layer fusion, precision calibration, kernel tuning. 2–5x speedup.               |
| **Triton**               | Open-source inference server. Multi-framework, dynamic batching, model versioning.                          |
| **NIM**                  | Pre-packaged, optimized LLM container. OpenAI-compatible API. Built on Triton + TensorRT-LLM.               |
| **NGC**                  | Catalog of GPU-optimized containers, models, Helm charts. Not a cloud provider.                             |
| **NVIDIA AI Enterprise** | Enterprise software platform with support SLAs, vGPU, certified containers.                                 |
| **NeMo**                 | End-to-end LLM framework: data curation → training → fine-tuning → evaluation → guardrails.                 |
| **Base Command Manager** | Cluster management for DGX POD/SuperPOD: provisioning, scheduling, monitoring.                              |

---

*Previous: [Module 2 — Inside an AI-Centric Data Center](Module-2-Inside-AI-Data-Center.md) | Next: [Module 4 — AI Workflows and Operations](Module-4-AI-Workflows-and-Operations.md)*
