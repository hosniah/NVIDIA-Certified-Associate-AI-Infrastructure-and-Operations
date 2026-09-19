# Module 2 — Inside an AI-Centric Data Center

## Learning Objectives

By the end of this module, you will be able to:

- Identify the four pillars of data center infrastructure: Compute, Networking, Storage, and Support Infrastructure
- Compare CPU, GPU, and DPU architectures and explain each processor's role
- Trace the evolution of GPUs from gaming to AI acceleration
- Explain data center power and cooling considerations, including PUE
- Describe data center networking topologies and the role of high-speed interconnects
- Understand storage requirements for AI workloads
- Explain the architecture of NVIDIA-Certified Servers

---

## 2.1 The Four Pillars of a Data Center

Every data center — whether it supports traditional enterprise IT or cutting-edge AI workloads — is built on four foundational pillars. An AI-centric data center places unique demands on each one.

### Pillar Overview

```
┌──────────────────────────────────────────────────────────────┐
│                    AI-Centric Data Center                     │
├──────────────┬──────────────┬──────────────┬────────────────┤
│   Compute    │  Networking  │   Storage    │    Support      │
│              │              │              │ Infrastructure  │
│  CPU         │  InfiniBand  │  Parallel    │  Power          │
│  GPU         │  Ethernet    │  File        │  Cooling        │
│  DPU         │  NVLink      │  Systems     │  Physical       │
│              │  NVSwitch    │  Object      │  Security       │
│              │              │  Storage     │                 │
└──────────────┴──────────────┴──────────────┴────────────────┘
```

**Compute** provides the processing power — CPUs for general tasks, GPUs for parallel AI workloads, and DPUs for infrastructure offload. **Networking** connects compute nodes and enables multi-node AI training across clusters. **Storage** feeds data to the GPUs fast enough to keep them utilized. **Support Infrastructure** ensures reliable power delivery, adequate cooling, and physical security.

---

## 2.2 Compute — CPU, GPU, and DPU

### The CPU (Central Processing Unit)

The CPU is the traditional workhorse of computing. It is a **general-purpose processor** optimized for sequential task execution with complex control logic.

**CPU Architecture Characteristics:**
- **Few powerful cores**: Modern server CPUs have 32–128 cores, each capable of independent complex operations
- **Large caches**: Multi-level cache hierarchy (L1, L2, L3) to minimize memory latency
- **Branch prediction**: Sophisticated logic to predict code execution paths and keep the pipeline full
- **Out-of-order execution**: Reorders instructions to maximize core utilization
- **Low latency**: Optimized for completing individual tasks quickly

**CPU Role in AI Data Centers:**
Even in GPU-accelerated systems, the CPU remains essential. It handles the operating system, orchestrates GPU workloads, manages data loading and preprocessing, runs the training framework (PyTorch, TensorFlow), and handles tasks that are inherently sequential.

**Market Leaders**: Intel Xeon Scalable, AMD EPYC, NVIDIA Grace (ARM-based)

### The GPU (Graphics Processing Unit)

The GPU is the engine of modern AI. Originally designed for rendering 3D graphics, its massively parallel architecture turned out to be ideal for the matrix operations at the heart of deep learning.

**GPU Architecture Characteristics:**
- **Thousands of simple cores**: An NVIDIA H100 has 16,896 CUDA cores and 528 Tensor Cores
- **SIMT execution model**: Single Instruction, Multiple Threads — thousands of threads execute the same operation on different data simultaneously
- **High-bandwidth memory (HBM)**: The H100 uses HBM3 providing up to 3.35 TB/s memory bandwidth
- **Tensor Cores**: Specialized hardware units that perform mixed-precision matrix multiply-accumulate operations (the core computation in deep learning)
- **High throughput**: Optimized for processing massive amounts of data in parallel

**The Car Analogy:**
Think of a CPU as a **sports car** — very fast, handles complex maneuvers, but carries only a few passengers. A GPU is a **bus** — slower per individual trip, but carries hundreds of passengers simultaneously. For AI workloads (where you have millions of simple, identical operations), the bus wins.

**CPU vs GPU Comparison:**

| Feature              | CPU                       | GPU                               |
| -------------------- | ------------------------- | --------------------------------- |
| **Core count**       | 32–128 cores              | 10,000+ CUDA cores                |
| **Core type**        | Complex, independent      | Simple, cooperative               |
| **Optimization**     | Low latency (single task) | High throughput (many tasks)      |
| **Memory bandwidth** | ~100–200 GB/s (DDR5)      | 2,000–8,000 GB/s (HBM)            |
| **Power draw**       | 200–400W                  | 300–1,000W                        |
| **Best for**         | Sequential logic, OS, I/O | Matrix math, parallel compute, AI |

### GPU History — From Gaming to AI

The GPU's journey from gaming peripheral to AI accelerator is a story of architectural serendipity combined with deliberate strategic investment by NVIDIA.

**Key Milestones:**

| Year     | Milestone                                       | Significance                                                                                                     |
| -------- | ----------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| **1999** | NVIDIA GeForce 256                              | First consumer GPU; hardware transform & lighting                                                                |
| **2006** | CUDA released                                   | NVIDIA opens GPU to general-purpose computing (GPGPU). Researchers can now write C code that runs on GPU cores   |
| **2009** | Stanford researchers use GPUs for deep learning | Early demonstrations that GPU parallelism dramatically accelerates neural network training                       |
| **2012** | AlexNet wins ImageNet                           | Deep CNN trained on 2 NVIDIA GTX 580 GPUs crushes the competition. This moment proved GPUs were the future of AI |
| **2016** | NVIDIA Pascal (P100)                            | First GPU designed specifically with deep learning in mind. HBM2 memory, NVLink interconnect                     |
| **2017** | NVIDIA Volta (V100)                             | Introduced **Tensor Cores** — dedicated matrix math units. A watershed moment for AI acceleration                |
| **2020** | NVIDIA Ampere (A100)                            | 3rd-gen Tensor Cores, TF32 precision, Multi-Instance GPU (MIG), structural sparsity                              |
| **2022** | NVIDIA Hopper (H100)                            | 4th-gen Tensor Cores, **Transformer Engine** with FP8 support, 3x AI performance over A100                       |
| **2024** | NVIDIA Blackwell (B200)                         | 5th-gen Tensor Cores, FP4 precision, 2nd-gen Transformer Engine, NVLink 5th gen at 1.8 TB/s                      |

### Beyond Moore's Law

Moore's Law — the observation that transistor density doubles approximately every two years — has been the backbone of computing progress for decades. However, physical limits (atomic-scale transistors, heat dissipation, power leakage) are slowing traditional scaling. The industry is responding with three strategies:

**1. Chiplets (Multi-Die Design)**
Instead of building one monolithic chip, manufacturers connect multiple smaller dies (chiplets) in a single package. Each chiplet can be optimized for its specific function and manufactured independently. AMD's EPYC processors and NVIDIA's Blackwell architecture use chiplet-based designs.

**2. 3D Stacking**
Stacking dies vertically (e.g., HBM memory stacking DRAM dies on top of each other connected by through-silicon vias) increases density without shrinking transistors further. HBM3e, used in the H200, stacks 8 or 12 DRAM dies.

**3. Domain-Specific Accelerators**
Rather than general-purpose transistor improvements, the industry builds specialized hardware for specific workloads: Tensor Cores for matrix math, Transformer Engines for attention operations, and DPUs for network processing. Performance gains come from architectural specialization rather than transistor scaling.

### The DPU (Data Processing Unit)

The DPU is the **third major processor** in modern data center architecture, alongside the CPU and GPU. It is a specialized processor designed to offload infrastructure tasks from the CPU, freeing CPU cycles for application workloads.

**What the DPU Offloads:**
- **Networking**: Packet processing, encryption/decryption, TCP/UDP offload, RDMA, network virtualization (VXLAN/Geneve overlay)
- **Storage**: NVMe-oF (NVMe over Fabrics), storage virtualization, compression, deduplication
- **Security**: Firewall, IDS/IPS, micro-segmentation, zero-trust architecture enforcement at the hardware level

**NVIDIA BlueField DPU:**
NVIDIA's DPU product line is called **BlueField**. The BlueField-3 DPU combines:
- ARM-based general-purpose cores for running infrastructure software
- Hardware accelerators for cryptography, regex, compression, and networking
- ConnectX-7 network adapter capabilities (up to 400 Gb/s)
- DOCA (Data Center Infrastructure on a Chip Architecture) — the software framework for programming BlueField DPUs

**Why DPUs Matter for AI:**
In an AI data center, every CPU cycle spent on networking overhead or storage management is a cycle *not* spent on feeding data to GPUs. By offloading these tasks to the DPU, the CPU can focus entirely on orchestrating GPU workloads, improving overall system efficiency.

### CPU, GPU, and DPU — The Three-Processor Architecture

| Processor | Primary Role                                        | Optimization                         | Analogy                                    |
| --------- | --------------------------------------------------- | ------------------------------------ | ------------------------------------------ |
| **CPU**   | General compute, orchestration                      | Latency (fast single tasks)          | The brain — makes decisions                |
| **GPU**   | Parallel compute, AI training/inference             | Throughput (many simultaneous tasks) | The muscle — does the heavy lifting        |
| **DPU**   | Infrastructure offload (network, storage, security) | I/O processing and data movement     | The nervous system — handles communication |

Modern NVIDIA-Certified Servers integrate all three processors, creating a balanced architecture where each processor handles what it does best.

---

## 2.3 Support Infrastructure — Power and Cooling

AI workloads place extraordinary demands on data center power and cooling infrastructure. A single modern GPU server can draw 10–15 kW — more than an entire traditional server rack consumed a decade ago.

### Power Considerations

**Thermal Design Power (TDP):**
TDP is the maximum amount of heat a component is designed to generate under sustained workload. For AI planning purposes:
- NVIDIA H100 SXM5 GPU: 700W TDP
- NVIDIA B200 GPU: ~1,000W TDP
- A single DGX H100 system (8 GPUs): ~10.2 kW total system power
- A single DGX B200 system (8 GPUs): ~14.3 kW total system power

**Power Infrastructure Chain:**
Utility power → Substation → Uninterruptible Power Supply (UPS) → Power Distribution Unit (PDU) → Server

Each step in this chain introduces conversion losses. A data center's electrical infrastructure must be designed to handle both the average load and peak demand, with redundancy (typically N+1 or 2N) to survive component failures.

### PUE — Power Usage Effectiveness

PUE is the industry-standard metric for measuring data center energy efficiency. It is defined as:

```
PUE = Total Facility Energy / IT Equipment Energy
```

**Interpreting PUE:**
- **PUE = 1.0**: Perfect efficiency (impossible in practice) — all energy goes to IT equipment
- **PUE = 1.2**: Excellent — for every 1.0W of IT load, only 0.2W is spent on cooling, lighting, and other overhead
- **PUE = 1.5**: Average for traditional data centers
- **PUE = 2.0**: Poor — half the energy is wasted on non-IT overhead

**Example Calculation:**
A data center consumes 10 MW of total power. The IT equipment (servers, storage, networking) consumes 8 MW. The remaining 2 MW goes to cooling, lighting, UPS losses, and other infrastructure.

PUE = 10 MW / 8 MW = **1.25**

**AI Data Centers and PUE:**
AI data centers face a PUE challenge: GPU-heavy workloads generate significantly more heat per rack than traditional servers, requiring more cooling capacity. Leading AI data centers achieve PUE values between 1.1 and 1.3 through advanced cooling techniques.

### Cooling Approaches

**Air Cooling:**
Traditional approach using CRAC (Computer Room Air Conditioning) units. Air flows through raised floors or overhead ducts, absorbs heat from servers, and returns to the cooling unit. Adequate for power densities up to approximately 15–20 kW per rack, but AI racks often exceed this.

**Liquid Cooling:**
As GPU power density increases, liquid cooling becomes essential:

| Method                          | How It Works                                                      | Use Case                    |
| ------------------------------- | ----------------------------------------------------------------- | --------------------------- |
| **Rear-Door Heat Exchangers**   | Liquid-cooled door replaces standard rear door; cools exhaust air | Retrofit for existing racks |
| **Direct-to-Chip (Cold Plate)** | Liquid flows through cold plates mounted directly on CPUs/GPUs    | DGX H100, DGX B200 systems  |
| **Immersion Cooling**           | Entire server submerged in dielectric fluid                       | Highest density deployments |

NVIDIA DGX B200 systems are designed for **direct liquid cooling**, with cold plates mounted directly on the GPUs and a facility water loop that carries the heat away. This is critical because the B200's ~14.3 kW system power simply cannot be cooled by air alone.

---

## 2.4 Data Center Networking

Networking is the circulatory system of an AI data center. For multi-GPU and multi-node AI training, the network determines how fast GPUs can exchange gradients, share model parameters, and synchronize — making it a critical factor in training performance.

### Network Types in a Data Center

A well-designed AI data center maintains **four separate networks**, each serving a distinct purpose:

**1. Compute Network (Data Plane)**
The high-speed network that carries AI training traffic between GPU nodes. This is where gradient synchronization, all-reduce operations, and model-parallel communication happen. Requires the highest bandwidth and lowest latency. Typically InfiniBand or high-performance Ethernet.

**2. Storage Network**
Connects compute nodes to storage systems. Must deliver sustained high throughput to keep GPU memory fed with training data. Often uses dedicated Ethernet, InfiniBand, or NVMe over Fabrics (NVMe-oF).

**3. In-Band Management Network**
The standard data network for SSH access, monitoring data, log collection, software updates, and general administration. Uses standard Ethernet (typically 1–25 GbE).

**4. Out-of-Band Management Network (OOB)**
A physically separate management network that provides access to hardware management controllers (BMC/IPMI/iDRAC/iLO) even when the server's OS is down. Used for remote power cycling, BIOS updates, and console access. Critical for lights-out operations.

### High-Speed Interconnects

#### InfiniBand

InfiniBand is a **high-performance, low-latency networking technology** that has been the standard for HPC and AI training clusters.

**Key Characteristics:**
- Designed for lossless, low-latency communication
- Supports **RDMA (Remote Direct Memory Access)**: data moves directly between the memory of two machines without involving the CPU or OS kernel — dramatically reducing latency
- Current generation: **NDR (Next Data Rate)** at 400 Gb/s per port
- Managed by a **Subnet Manager (OpenSM)** that configures routing and manages the fabric
- Credit-based flow control ensures zero packet loss

**NVIDIA InfiniBand Products:**
- **ConnectX-7**: Network adapter (NIC/HCA) supporting NDR 400 Gb/s InfiniBand
- **Quantum-2 switches**: InfiniBand switches with up to 64 ports of NDR 400 Gb/s

#### Ethernet for AI (Spectrum-X)

While InfiniBand has traditionally dominated AI networking, NVIDIA's **Spectrum-X** platform brings AI-optimized Ethernet that narrows the performance gap:
- Built on **Spectrum-4 switches** + **BlueField-3 DPUs**
- Provides **RoCE (RDMA over Converged Ethernet)** for RDMA capabilities over standard Ethernet infrastructure
- Adaptive routing and congestion control optimized for AI traffic patterns
- Appealing for organizations that want to use their existing Ethernet infrastructure and expertise

**InfiniBand vs. Ethernet for AI:**

| Feature        | InfiniBand (NDR)                 | Spectrum-X (Ethernet)           |
| -------------- | -------------------------------- | ------------------------------- |
| **Bandwidth**  | 400 Gb/s                         | 400 Gb/s                        |
| **Latency**    | ~1 μs                            | ~2–3 μs                         |
| **RDMA**       | Native (built-in)                | RoCE (overlay)                  |
| **Lossless**   | Yes (credit-based)               | Yes (PFC/ECN)                   |
| **Best for**   | Large-scale AI training clusters | Mixed AI + enterprise workloads |
| **Management** | Subnet Manager (OpenSM)          | Standard network management     |

### Intra-Node Interconnects

#### NVLink

NVLink is NVIDIA's proprietary **GPU-to-GPU interconnect** within a single server node. It provides dramatically higher bandwidth than PCIe for GPU-to-GPU communication.

**NVLink Evolution:**

| Generation | Bandwidth (bidirectional per GPU) | Introduced With  |
| ---------- | --------------------------------- | ---------------- |
| NVLink 1.0 | 160 GB/s                          | Pascal (P100)    |
| NVLink 2.0 | 300 GB/s                          | Volta (V100)     |
| NVLink 3.0 | 600 GB/s                          | Ampere (A100)    |
| NVLink 4.0 | 900 GB/s                          | Hopper (H100)    |
| NVLink 5.0 | 1,800 GB/s (1.8 TB/s)             | Blackwell (B200) |

For comparison, PCIe Gen5 provides approximately 64 GB/s per x16 slot — NVLink 4.0 is over **14x faster**.

#### NVSwitch

NVSwitch is a dedicated switch chip that provides **all-to-all NVLink connectivity** between GPUs within a node. Without NVSwitch, GPUs would need point-to-point NVLink connections (limiting topology). With NVSwitch, every GPU can communicate directly with every other GPU at full NVLink bandwidth.

**DGX H100 NVSwitch Configuration:**
- 4 NVSwitch chips per system
- Connects all 8 H100 GPUs in an all-to-all topology
- Total bisection bandwidth: 3.6 TB/s

#### NVLink-C2C (Chip-to-Chip)

NVLink-C2C is a coherent chip-to-chip interconnect used in the **Grace Hopper Superchip**, connecting the Grace ARM CPU directly to the Hopper GPU. It provides 900 GB/s of coherent bandwidth, allowing the CPU and GPU to share a unified memory address space — eliminating the traditional bottleneck of copying data between CPU and GPU memory over PCIe.

### Network Topology

AI data center networks typically use a **fat-tree (Clos) topology** or a **spine-leaf architecture**:

```
         ┌──────┐  ┌──────┐  ┌──────┐
         │Spine │  │Spine │  │Spine │
         │Switch│  │Switch│  │Switch│
         └──┬───┘  └──┬───┘  └──┬───┘
            │         │         │
    ┌───────┼─────────┼─────────┼───────┐
    │       │         │         │       │
┌───┴──┐┌──┴───┐┌────┴──┐┌────┴──┐┌──┴───┐
│ Leaf ││ Leaf  ││ Leaf  ││ Leaf  ││ Leaf │
│Switch││Switch ││Switch ││Switch ││Switch│
└──┬───┘└──┬───┘└───┬───┘└───┬───┘└──┬───┘
   │       │        │        │       │
 Servers  Servers  Servers  Servers  Servers
```

**Key properties:**
- Every leaf switch connects to every spine switch
- Non-blocking: any server can communicate with any other server at full bandwidth
- Scales horizontally by adding more spine and leaf switches
- Consistent latency: all server-to-server paths traverse the same number of hops

---

## 2.5 Storage for AI Workloads

AI training workloads are data-hungry. A large language model training run may consume petabytes of text data, and computer vision training may require millions of high-resolution images. The storage system must deliver data to GPUs fast enough to keep them fully utilized — if the GPUs are waiting for data, expensive compute capacity is being wasted.

### Storage Requirements for AI

| Requirement         | Why It Matters                                                                                  |
| ------------------- | ----------------------------------------------------------------------------------------------- |
| **High throughput** | Multiple GPUs reading training data simultaneously need aggregate bandwidth of 10s–100s of GB/s |
| **Low latency**     | Checkpoint saving and loading during training must be fast to minimize downtime                 |
| **Large capacity**  | Training datasets can range from terabytes to petabytes                                         |
| **Parallel access** | Hundreds of GPU nodes must read from the same dataset concurrently                              |

### Storage Technologies

**Parallel File Systems:**
Parallel file systems stripe data across many storage servers, allowing hundreds of clients to read and write simultaneously at high aggregate throughput.
- **Lustre**: Open-source parallel file system widely used in HPC and AI. Scales to thousands of clients and exabytes of storage
- **GPFS/IBM Spectrum Scale**: Enterprise parallel file system with strong POSIX compliance and data management features
- **BeeGFS**: High-performance parallel file system designed for ease of deployment

**Object Storage:**
For large-scale unstructured data (images, video, documents), object storage provides scalable, cost-effective capacity:
- **MinIO**: S3-compatible object storage popular in AI pipelines
- **Ceph**: Open-source distributed storage supporting object, block, and file interfaces

**NVMe and All-Flash:**
Local NVMe SSDs in compute nodes provide the lowest-latency access for training data caching and checkpoint storage. All-flash NVMe arrays (like VAST Data, Pure Storage, NetApp) are increasingly used for AI storage tiers.

### GPUDirect Storage (GDS)

NVIDIA GPUDirect Storage creates a **direct data path between storage and GPU memory**, bypassing the CPU entirely. In traditional architectures, data must be read from storage into CPU memory (system RAM), then copied to GPU memory — introducing latency and consuming CPU resources and PCIe bandwidth.

With GPUDirect Storage:
```
Traditional:  Storage → CPU Memory → GPU Memory  (two copies, CPU involved)
GDS:          Storage → GPU Memory                (one copy, CPU bypassed)
```

This is particularly valuable for AI training workloads where the GPU needs to ingest large volumes of training data as quickly as possible.

### GPUDirect RDMA

GPUDirect RDMA enables **direct data transfer between a network adapter and GPU memory** without going through CPU memory. This is critical for multi-node AI training where GPUs on different servers need to exchange gradient updates during the all-reduce synchronization step.

```
Traditional:  GPU → CPU Memory → Network → CPU Memory → GPU  (on another node)
GPUDirect RDMA: GPU → Network → GPU  (direct, bypasses CPU memory on both sides)
```

---

## 2.6 NVIDIA-Certified Servers

NVIDIA-Certified Servers are systems from OEM partners (Dell, HPE, Lenovo, Supermicro, and others) that have been **validated and certified by NVIDIA** to run GPU-accelerated workloads optimally. Certification ensures that the server's hardware configuration, firmware, BIOS settings, drivers, and cooling are all optimized for NVIDIA GPUs.

### Three-Processor Architecture

A modern NVIDIA-Certified Server integrates all three processor types:

```
┌──────────────────────────────────────────────────────────┐
│                  NVIDIA-Certified Server                   │
│                                                           │
│  ┌─────────┐    ┌─────────────────────────┐   ┌───────┐ │
│  │  CPU    │    │     GPU (×2, ×4, or ×8)  │   │  DPU  │ │
│  │         │◄──►│                          │   │       │ │
│  │ Intel   │PCIe│  NVIDIA A100/H100/B200  │   │ Blue- │ │
│  │ or AMD  │    │  Connected via NVLink    │   │ Field │ │
│  │ or Grace│    │  + NVSwitch             │   │       │ │
│  └─────────┘    └─────────────────────────┘   └───────┘ │
│       ▲                    ▲                      ▲      │
│       │                    │                      │      │
│    General              AI Training           Network,   │
│    compute,             & Inference           Storage,   │
│    orchestration                              Security   │
└──────────────────────────────────────────────────────────┘
```

### Certification Benefits

- **Validated performance**: Hardware, drivers, and firmware tested together for optimal GPU utilization
- **Enterprise support**: Joint support from NVIDIA and the OEM partner
- **Software compatibility**: Certified to run NVIDIA AI Enterprise, CUDA, and the full NVIDIA software stack
- **Deployment confidence**: Eliminates guesswork in system configuration

### Form Factors

NVIDIA-Certified Servers come in various configurations based on GPU count and deployment needs:
- **1–2 GPU servers**: Inference, edge AI, entry-level training
- **4 GPU servers**: Mid-range training, multi-model inference
- **8 GPU servers (DGX-class)**: Large-scale training, maximum GPU density with NVLink/NVSwitch interconnect

---

## Module 2 Summary

| Concept                      | Key Takeaway                                                                                                          |
| ---------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| **Four Pillars**             | Compute, Networking, Storage, Support Infrastructure                                                                  |
| **CPU vs GPU**               | CPU = latency-optimized, few complex cores; GPU = throughput-optimized, thousands of simple cores                     |
| **DPU**                      | Offloads networking, storage, and security from CPU. NVIDIA BlueField.                                                |
| **GPU Evolution**            | Gaming (1999) → CUDA (2006) → AlexNet (2012) → Tensor Cores (2017) → Transformer Engine (2022) → FP4/Blackwell (2024) |
| **PUE**                      | Power Usage Effectiveness = Total Facility Energy / IT Equipment Energy. Lower is better.                             |
| **Cooling**                  | Air cooling insufficient for modern GPU density. Direct liquid cooling (cold plate) required for DGX systems.         |
| **Compute Network**          | InfiniBand (lowest latency, native RDMA) or Spectrum-X Ethernet (RoCE)                                                |
| **NVLink**                   | GPU-to-GPU interconnect. H100: 900 GB/s. B200: 1.8 TB/s. Orders of magnitude faster than PCIe.                        |
| **NVSwitch**                 | All-to-all GPU fabric within a node                                                                                   |
| **Storage**                  | Parallel file systems (Lustre, GPFS), GPUDirect Storage bypasses CPU                                                  |
| **NVIDIA-Certified Servers** | OEM servers validated for optimal GPU workload performance. CPU + GPU + DPU architecture.                             |

---

*Previous: [Module 1 — AI Fundamentals](Module-1-AI-Fundamentals.md) | Next: [Module 3 — NVIDIA Technology Stack](Module-3-NVIDIA-Technology-Stack.md)*
