# Module 4 — AI Workflows and Operations

## Learning Objectives

By the end of this module, you will be able to:

- Describe the end-to-end AI/ML pipeline from data preparation through deployment
- Differentiate between training and inference workloads and their infrastructure requirements
- Explain distributed training strategies: data parallelism, model parallelism, and pipeline parallelism
- Describe inference optimization techniques and deployment patterns
- Use NVIDIA monitoring and management tools: nvidia-smi, DCGM, and Base Command Manager
- Explain containerized GPU workloads with the NVIDIA Container Toolkit
- Understand GPU orchestration with Kubernetes GPU Operator and Slurm
- Configure Multi-Instance GPU (MIG) and vGPU for multi-tenant environments

---

## 4.1 The AI/ML Pipeline

Building and deploying an AI model is not a single step — it is a multi-stage pipeline. Each stage has distinct computational requirements and uses different tools from the NVIDIA stack.

### Pipeline Stages

```
┌────────┐   ┌────────┐   ┌──────────┐   ┌────────────┐   ┌──────────┐   ┌────────────┐
│  Data  │──►│  Data  │──►│  Model   │──►│  Model     │──►│ Deploy-  │──►│ Monitoring │
│Collect.│   │ Prep.  │   │ Training │   │ Evaluation │   │  ment    │   │ & Ops      │
└────────┘   └────────┘   └──────────┘   └────────────┘   └──────────┘   └────────────┘
                                                                │
                                                                │ Feedback loop
                                                                ▼
                                                          ┌────────────┐
                                                          │  Retrain   │
                                                          └────────────┘
```

### Stage 1: Data Collection and Curation

AI models are only as good as their training data. This stage involves gathering, cleaning, and curating datasets.

**Activities:**
- Collecting raw data from various sources (databases, APIs, sensors, web scraping, public datasets)
- Data deduplication and decontamination
- Quality filtering and annotation
- Data labeling (for supervised learning)
- Compliance and privacy filtering (removing PII, applying data governance policies)

**NVIDIA Tools:**
- **NeMo Curator**: Automated data curation pipeline for LLM training data — deduplication, quality filtering, text extraction, and PII detection at scale using GPU acceleration

### Stage 2: Data Preparation and Preprocessing

Raw data must be transformed into a format suitable for model training.

**Activities:**
- Feature engineering (creating meaningful input variables from raw data)
- Normalization and standardization
- Tokenization (for text data: converting text to numerical token sequences)
- Image augmentation (random cropping, flipping, color jittering) to increase training data diversity
- Train/validation/test split

**NVIDIA Tools:**
- **RAPIDS cuDF**: GPU-accelerated DataFrame operations for tabular data preprocessing
- **RAPIDS cuML**: GPU-accelerated feature engineering and traditional ML algorithms
- **DALI (Data Loading Library)**: GPU-accelerated image decoding, augmentation, and preprocessing pipeline

### Stage 3: Model Training

Training is the most compute-intensive stage — it is where the model learns patterns from the data by iteratively adjusting its parameters to minimize a loss function.

**Training Loop (simplified):**
1. Load a batch of training data into GPU memory
2. Forward pass: compute the model's predictions
3. Compute the loss (difference between predictions and ground truth)
4. Backward pass: compute gradients (how to adjust each parameter to reduce loss)
5. Update model parameters using an optimizer (SGD, Adam, etc.)
6. Repeat for millions of iterations

**Computational Requirements:**
- Training a large language model can require thousands of GPUs running for weeks or months
- GPU memory (HBM) must be large enough to hold the model, optimizer states, gradients, and activations
- High-bandwidth interconnects (NVLink, InfiniBand) are critical for multi-GPU gradient synchronization

### Stage 4: Model Evaluation

After training, the model is evaluated on held-out data to assess its performance.

**Common Metrics:**
- **Accuracy**: Percentage of correct predictions (classification)
- **Precision/Recall/F1**: More nuanced classification metrics
- **Perplexity**: How well a language model predicts a sequence (lower is better)
- **BLEU/ROUGE**: Machine translation and summarization quality
- **Human evaluation**: For generative AI, human raters assess quality, helpfulness, and safety

### Stage 5: Model Optimization and Deployment

The trained model must be optimized for production inference and deployed to serve predictions.

**Optimization Techniques (covered in Section 4.3):**
- Quantization (reducing precision from FP32 to FP16/INT8/FP8/FP4)
- Layer fusion (combining operations)
- Pruning (removing unnecessary model parameters)
- Knowledge distillation (training a smaller model to mimic a larger one)

**Deployment Patterns:**
- Cloud inference (GPU servers in data center)
- Edge inference (NVIDIA Jetson, L4 GPUs at the edge)
- Hybrid (training in cloud, inference at edge)

### Stage 6: Monitoring and Operations

Deployed models require ongoing monitoring and management (covered in detail in Section 4.5).

---

## 4.2 Distributed Training

When a model is too large for a single GPU — or when training would take too long on a single GPU — the workload must be distributed across multiple GPUs. There are three primary strategies.

### Data Parallelism

**Concept:** The model is replicated on each GPU. Each GPU processes a different batch of training data independently. After each iteration, gradients are synchronized (aggregated) across all GPUs using an **all-reduce** operation.

```
GPU 0: Model Copy → Batch 0 → Gradients ──┐
GPU 1: Model Copy → Batch 1 → Gradients ──┤── All-Reduce ──► Updated Model
GPU 2: Model Copy → Batch 2 → Gradients ──┤    (via NCCL)
GPU 3: Model Copy → Batch 3 → Gradients ──┘
```

**When to Use:** The model fits in a single GPU's memory, but you want to process data faster.

**Communication:** All-reduce is the dominant communication pattern. NCCL optimizes this based on the GPU topology.

**Scaling:** Near-linear speedup with more GPUs (limited by communication overhead during all-reduce). The global batch size increases with GPU count.

### Model Parallelism (Tensor Parallelism)

**Concept:** The model itself is split across GPUs — each GPU holds a portion of the model's parameters. A single input flows through GPUs sequentially or in parallel, with intermediate results passed between GPUs.

```
Input ──► GPU 0 (Layers 1-4) ──► GPU 1 (Layers 5-8) ──► Output
          └── Model split across GPUs ──┘
```

**Tensor Parallelism (TP)** specifically splits individual layers across GPUs. For example, a large matrix multiplication in a Transformer's attention layer can be split so each GPU computes part of the result.

**When to Use:** The model is too large to fit in a single GPU's memory (e.g., a 70B-parameter LLM).

**Communication:** Requires high-bandwidth, low-latency interconnects (NVLink) because intermediate activations must be exchanged between GPUs at every layer.

### Pipeline Parallelism

**Concept:** The model is split into sequential stages, with each stage assigned to a different GPU. Multiple micro-batches are processed simultaneously — as GPU 0 finishes processing micro-batch 1 and starts micro-batch 2, GPU 1 begins processing micro-batch 1.

```
Time →
GPU 0: [Micro-batch 1] [Micro-batch 2] [Micro-batch 3] ...
GPU 1:                  [Micro-batch 1] [Micro-batch 2] ...
GPU 2:                                  [Micro-batch 1] ...
```

**When to Use:** Very large models where each stage fits on one GPU, and you want to maximize GPU utilization by overlapping computation.

**Trade-off:** Pipeline "bubbles" (idle time at the beginning and end of a pipeline) reduce efficiency. Various scheduling strategies (1F1B, interleaved) minimize bubble overhead.

### Combining Strategies (3D Parallelism)

Large-scale AI training typically combines all three strategies — this is called **3D parallelism**:
- **Tensor Parallelism** within a single node (across NVLink-connected GPUs)
- **Pipeline Parallelism** across nodes (across InfiniBand)
- **Data Parallelism** across groups of nodes

This combination allows training models with hundreds of billions of parameters across thousands of GPUs.

---

## 4.3 Inference Optimization

Inference (using a trained model to make predictions on new data) has fundamentally different characteristics than training.

### Training vs. Inference

| Characteristic      | Training                                | Inference                                |
| ------------------- | --------------------------------------- | ---------------------------------------- |
| **Goal**            | Learn model parameters                  | Generate predictions                     |
| **Compute pattern** | Large batch, high throughput            | Often single request, low latency        |
| **GPU utilization** | Typically high (100%)                   | Often low without optimization           |
| **Precision**       | FP32 or mixed (FP16/BF16)               | Can use lower precision (INT8, FP8, FP4) |
| **Key metric**      | Time to convergence                     | Latency and throughput                   |
| **GPU memory**      | Must hold model + gradients + optimizer | Only model + activations (smaller)       |
| **Scaling driver**  | Dataset size, model size                | Request volume, latency requirements     |

### Optimization Techniques

**Quantization:**
Reducing the numerical precision of model parameters from FP32 to lower-precision formats. Each reduction in precision roughly doubles throughput:
- FP32 → FP16: 2x throughput (most models tolerate this with zero accuracy loss)
- FP16 → INT8: 2x additional throughput (requires calibration to maintain accuracy)
- INT8 → FP4: 2x additional throughput (Blackwell-generation GPUs)

TensorRT automates precision calibration — it measures the accuracy impact of quantization on a calibration dataset and applies the most aggressive quantization that stays within the accuracy tolerance.

**Layer and Tensor Fusion:**
Combining multiple sequential operations into a single GPU kernel to reduce memory reads/writes and kernel launch overhead. Example: instead of executing ReLU, Bias Add, and Convolution as three separate operations, TensorRT fuses them into one.

**Dynamic Batching:**
Triton Inference Server can wait a short time to accumulate multiple incoming requests into a single batch before sending them to the GPU. This dramatically improves GPU utilization and throughput at the cost of small additional latency.

**KV-Cache Optimization:**
For autoregressive LLM inference (generating text token by token), the Key-Value cache stores intermediate attention computations so they don't need to be recomputed for each new token. Efficient KV-cache management is critical for LLM serving performance.

---

## 4.4 Checkpointing and Fault Tolerance

AI training runs that span weeks across thousands of GPUs are vulnerable to hardware failures. Checkpointing is the mechanism that provides fault tolerance.

### What Is a Checkpoint?

A checkpoint is a snapshot of the complete training state saved to persistent storage:
- Model parameters (weights and biases)
- Optimizer state (momentum, adaptive learning rate parameters)
- Learning rate scheduler state
- Training iteration count
- Random number generator states

### Checkpoint Strategy

**Frequency:** Checkpoints are saved periodically (e.g., every 1,000 iterations or every hour). The frequency balances two costs: the time spent writing checkpoints vs. the time lost repeating work if a failure occurs between checkpoints.

**Storage:** Checkpoints are large (a 70B-parameter model checkpoint can exceed 500 GB including optimizer state). They require high-throughput parallel storage (Lustre, GPFS) to minimize the time training is paused during checkpoint writes.

**Recovery:** When a hardware failure occurs, training resumes from the most recent checkpoint. All work since the last checkpoint is lost, but the run does not need to restart from scratch.

---

## 4.5 AI Operations — Monitoring and Management

Operating an AI data center requires specialized tools for monitoring GPU health, managing workloads, and ensuring infrastructure reliability. This section covers the operational tools that make up 22% of the NCA-AIIO exam.

### nvidia-smi (System Management Interface)

**nvidia-smi** is the foundational command-line tool for monitoring NVIDIA GPUs on a single node. It reports:
- GPU utilization percentage
- GPU memory usage (used / total)
- GPU temperature
- Power draw (current / TDP limit)
- Running processes and their GPU memory consumption
- Driver and CUDA version
- ECC error counts

**Limitations:**
- Single-node only — no cluster-wide view
- Point-in-time snapshot — no historical data
- No native alerting or integration with monitoring systems
- No automated health checks

nvidia-smi is essential for ad-hoc debugging and quick checks, but production AI operations require DCGM.

### DCGM (Data Center GPU Manager)

**DCGM** is NVIDIA's enterprise-grade GPU monitoring and management tool designed for data center-scale deployments. It provides capabilities far beyond nvidia-smi:

**Monitoring Metrics:**
- **SM (Streaming Multiprocessor) utilization**: How busy the GPU's compute units are
- **Memory bandwidth utilization**: How much of the GPU's memory bandwidth is being used
- **Tensor Core utilization**: How actively the Tensor Cores are being used (critical for AI workloads)
- **NVLink throughput**: Data transfer rate across NVLink connections
- **PCIe throughput**: Data transfer rate on the PCIe bus
- **Power draw**: Current power consumption
- **Temperature**: GPU die temperature and memory temperature
- **ECC errors**: Correctable and uncorrectable memory errors (indicates potential hardware failure)
- **Thermal throttling**: Whether the GPU is reducing clock speed due to thermal limits

**Key Capabilities:**
- **Health checks**: Automated diagnostic tests that detect GPU hardware problems proactively
- **Policy-based monitoring**: Define thresholds and trigger actions when metrics exceed limits
- **Group management**: Monitor and manage GPUs in logical groups
- **Prometheus/Grafana integration**: Export metrics via the DCGM exporter for visualization in Grafana dashboards and alerting via Prometheus Alertmanager
- **Job-level statistics**: Track GPU utilization per training job, not just per GPU

**Architecture:**
DCGM runs as an agent (daemon) on each GPU node. A centralized management layer collects data from all agents, providing a cluster-wide view of GPU health and utilization.

### NVML (NVIDIA Management Library)

**NVML** is the C-based API that underlies both nvidia-smi and DCGM. It provides programmatic access to GPU monitoring and management functions. DCGM is built on top of NVML with additional features (health checks, policy management, Prometheus export). Custom monitoring integrations can also use NVML directly.

---

## 4.6 Containerized GPU Workloads

Containers are the standard deployment unit for AI workloads. They provide reproducibility (same environment everywhere), isolation (workloads don't interfere with each other), and portability (run on any system with the container runtime).

### NVIDIA Container Toolkit

The **NVIDIA Container Toolkit** (formerly nvidia-docker) enables containers to access NVIDIA GPUs. It provides:
- A custom container runtime that makes GPUs visible inside containers
- Automatic GPU driver injection into containers (the container doesn't need to include the GPU driver)
- Support for Docker, containerd, CRI-O, and Podman

**Usage:**
```bash
# Run a GPU-accelerated container
docker run --gpus all nvidia/cuda:12.0-base nvidia-smi

# Allocate specific GPUs
docker run --gpus '"device=0,1"' my-training-image python train.py
```

The `--gpus` flag tells Docker to use the NVIDIA runtime, which makes the specified GPUs available inside the container. Without the NVIDIA Container Toolkit, containers cannot see or use GPUs.

### NGC Containers

NGC provides pre-built, GPU-optimized containers for all major AI frameworks and tools:
- `nvcr.io/nvidia/pytorch:24.05-py3` — PyTorch with CUDA, cuDNN, NCCL, and TensorRT pre-installed
- `nvcr.io/nvidia/tensorflow:24.05-tf2-py3` — TensorFlow with full GPU support
- `nvcr.io/nvidia/tritonserver:24.05-py3` — Triton Inference Server

These containers are tested and validated on NVIDIA GPUs. Using NGC containers instead of building your own eliminates compatibility issues between drivers, CUDA versions, and framework versions.

---

## 4.7 GPU Orchestration — Kubernetes and Slurm

Managing GPU resources at scale requires workload orchestration — automatically scheduling GPU workloads, managing resource allocation, and handling failures.

### NVIDIA GPU Operator (Kubernetes)

The **GPU Operator** automates the deployment and management of all NVIDIA software components needed to run GPU workloads on Kubernetes. It uses the Kubernetes Operator pattern to manage:

**Components Managed by GPU Operator:**
- NVIDIA GPU drivers
- NVIDIA Container Toolkit
- NVIDIA Device Plugin (makes GPUs visible as schedulable Kubernetes resources)
- DCGM and DCGM Exporter (GPU monitoring)
- GPU Feature Discovery (labels nodes with GPU capabilities)
- MIG Manager (for MIG-partitioned GPUs)

**Without GPU Operator:** Administrators must manually install and maintain GPU drivers, container toolkit, device plugins, and monitoring tools on every Kubernetes node — and keep all versions in sync. Any update requires manual intervention across the cluster.

**With GPU Operator:** All components are deployed automatically via Helm charts and managed as Kubernetes custom resources. Updates are rolling and automated. New nodes are GPU-ready within minutes of joining the cluster.

**GPU Scheduling in Kubernetes:**
Once the GPU Operator is deployed, users request GPUs in their pod specifications:

```yaml
resources:
  limits:
    nvidia.com/gpu: 2   # Request 2 GPUs
```

Kubernetes schedules the pod on a node with at least 2 available GPUs.

### Slurm Workload Manager

**Slurm** is the dominant workload manager in HPC (High-Performance Computing) environments. Many large-scale AI training clusters use Slurm rather than Kubernetes because:
- Mature support for multi-node, multi-GPU jobs
- Tight integration with InfiniBand and MPI (Message Passing Interface)
- Strong scheduling policies for large, long-running training jobs
- Gang scheduling (ensuring all GPUs for a job start simultaneously)

**Slurm Concepts:**
- **Partition**: A logical group of nodes (similar to a Kubernetes namespace)
- **Job**: A user-submitted workload requesting specific resources (GPUs, CPUs, memory, time)
- **GRES (Generic Resources)**: How Slurm tracks GPUs as schedulable resources

**Slurm + NVIDIA Integration:**
Slurm uses the GRES plugin to track GPU resources. Administrators configure GPU types and counts per node, and users request specific GPUs in their job scripts.

### Kubernetes vs. Slurm for AI

| Feature         | Kubernetes + GPU Operator                         | Slurm                                        |
| --------------- | ------------------------------------------------- | -------------------------------------------- |
| **Origin**      | Cloud-native container orchestration              | HPC workload scheduling                      |
| **Best for**    | Inference serving, microservices, mixed workloads | Large-scale distributed training             |
| **GPU support** | Via GPU Operator + Device Plugin                  | Via GRES plugin                              |
| **Network**     | Standard Kubernetes networking (Ethernet)         | Direct InfiniBand integration with MPI       |
| **Scheduling**  | Pod-level                                         | Gang scheduling (all nodes at once)          |
| **Ecosystem**   | Cloud-native (Helm, operators, service mesh)      | HPC (MPI, module system, shared filesystems) |

Many organizations use both: Slurm for training, Kubernetes for inference and serving.

---

## 4.8 GPU Partitioning — MIG and vGPU

Modern NVIDIA GPUs are powerful enough that a single GPU is often more than a single workload needs — especially for inference. GPU partitioning technologies allow multiple workloads to share a single physical GPU.

### Multi-Instance GPU (MIG)

**MIG** is a hardware-level GPU partitioning technology available on A100, H100, and newer NVIDIA GPUs. It divides a single GPU into up to **7 isolated instances**, each with its own:
- Compute resources (Streaming Multiprocessors)
- Memory (dedicated HBM slice)
- Memory bandwidth
- L2 cache

**Key Properties:**
- **Hardware isolation**: MIG instances are fully isolated at the hardware level — one instance cannot see or interfere with another. This provides predictable performance and is suitable for multi-tenant environments.
- **Fault isolation**: An ECC error in one MIG instance does not affect other instances.
- **Simultaneous use**: All MIG instances on a GPU operate concurrently.

**MIG Profile Math:**
A GPU has a fixed number of compute slices (up to 7) and memory slices (up to 8). Both totals must hold simultaneously when defining MIG profiles:

**H100 MIG Profiles Example:**

| Profile | Compute Slices | Memory | Use Case                           |
| ------- | -------------- | ------ | ---------------------------------- |
| 1g.10gb | 1/7 of GPU     | 10 GB  | Small inference model              |
| 2g.20gb | 2/7 of GPU     | 20 GB  | Medium inference                   |
| 3g.40gb | 3/7 of GPU     | 40 GB  | Larger inference or development    |
| 4g.40gb | 4/7 of GPU     | 40 GB  | Balanced training/inference        |
| 7g.80gb | 7/7 of GPU     | 80 GB  | Full GPU (MIG disabled equivalent) |

**MIG + Kubernetes:**
The GPU Operator's MIG Manager automatically creates and manages MIG instances on Kubernetes nodes. Each MIG instance appears as a separate schedulable GPU resource.

### vGPU (Virtual GPU)

**vGPU** is NVIDIA's GPU virtualization technology that allows a physical GPU to be shared across multiple virtual machines (VMs). It is part of **NVIDIA AI Enterprise**.

**vGPU vs MIG:**

| Feature               | MIG                            | vGPU                                  |
| --------------------- | ------------------------------ | ------------------------------------- |
| **Isolation level**   | Hardware-partitioned           | Time-sliced or MIG-backed             |
| **Workload type**     | Containers, bare metal         | Virtual machines                      |
| **GPU architectures** | A100, H100, B200+              | Broader GPU support                   |
| **Management**        | nvidia-smi, GPU Operator       | NVIDIA vGPU Manager + hypervisor      |
| **License**           | Free (built into GPU hardware) | Requires NVIDIA AI Enterprise license |
| **Use case**          | Cloud/container multi-tenancy  | VMware/KVM virtualized environments   |

**vGPU Scheduling Modes:**
- **Time-sliced**: The GPU's compute resources are time-shared between VMs (similar to CPU time-sharing). Simpler but less predictable performance.
- **MIG-backed**: Each VM gets a dedicated MIG instance with hardware isolation. Best of both worlds — virtualization management with hardware-level isolation.

---

## 4.9 Cloud, On-Premises, and Hybrid Deployments

AI infrastructure can be deployed in various models, each with trade-offs:

### On-Premises

**Advantages:**
- Full control over hardware and data (critical for regulated industries)
- No data sovereignty concerns — data never leaves your facility
- Predictable costs for sustained, high-utilization workloads
- Maximum performance (no virtualization overhead, direct hardware access)

**Challenges:**
- Large upfront capital expenditure (CAPEX)
- Long lead times for hardware procurement
- Requires in-house data center expertise (power, cooling, networking)
- Difficult to scale quickly

### Cloud

**Advantages:**
- No upfront capital expenditure — pay as you go (OPEX)
- Rapid scaling — provision GPU instances in minutes
- Managed infrastructure — the cloud provider handles hardware maintenance
- Geographic distribution for inference at the edge

**Cloud GPU Providers:**
- **NVIDIA DGX Cloud**: NVIDIA-hosted DGX infrastructure available through cloud partners
- **AWS (EC2 P5)**: H100-based instances
- **Google Cloud (A3)**: H100-based instances
- **Microsoft Azure (ND H100 v5)**: H100-based instances
- **Oracle Cloud (BM.GPU.H100)**: Bare-metal H100 instances

**Challenges:**
- Higher cost for sustained, long-running workloads
- Data transfer costs (egress charges)
- Less control over hardware configuration
- Potential for GPU availability constraints during peak demand

### Hybrid

Many organizations adopt a **hybrid strategy**:
- On-premises DGX/HGX clusters for sensitive data and sustained training workloads
- Cloud GPU instances for burst capacity, experimentation, and global inference deployment
- **NVIDIA Base Command Platform** can manage workloads across both on-premises and cloud GPU resources

---

## 4.10 MLOps — Operationalizing AI

MLOps (Machine Learning Operations) applies DevOps principles to the AI/ML lifecycle, ensuring models are developed, deployed, and maintained reliably and at scale.

### MLOps Practices

**Version Control:**
- **Data versioning**: Tracking which version of the training dataset was used for each model
- **Model versioning**: Maintaining a registry of trained models with their hyperparameters, metrics, and lineage
- **Code versioning**: Standard git-based version control for training scripts and configurations

**Continuous Training:**
- Automated retraining pipelines that trigger when new data is available or model performance degrades
- Automated evaluation gates that prevent deploying models that don't meet performance thresholds

**Model Registry:**
A central repository for trained models, their metadata (training data, hyperparameters, evaluation metrics), and deployment status. NGC serves as NVIDIA's model registry.

**Monitoring in Production:**
- **Data drift**: Detecting when incoming data distribution shifts away from the training data
- **Model drift**: Detecting when model accuracy degrades over time
- **Infrastructure monitoring**: DCGM metrics for GPU health and utilization

**A/B Testing and Canary Deployments:**
Triton Inference Server supports model versioning, enabling gradual rollout of new model versions:
- Route a small percentage of traffic to the new model
- Compare latency and accuracy metrics
- Gradually increase traffic if the new model performs well

---

## Module 4 Summary

| Concept                      | Key Takeaway                                                                                                                  |
| ---------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| **AI Pipeline**              | Data Collection → Preprocessing → Training → Evaluation → Optimization → Deployment → Monitoring                              |
| **Data Parallelism**         | Model replicated on each GPU; data split across GPUs; gradients synchronized via all-reduce (NCCL)                            |
| **Model/Tensor Parallelism** | Model split across GPUs; requires high-bandwidth NVLink; used when model doesn't fit in one GPU                               |
| **Pipeline Parallelism**     | Model split into stages; micro-batches overlap execution; trade-off is "bubble" idle time                                     |
| **3D Parallelism**           | Combining TP + PP + DP for training the largest models across thousands of GPUs                                               |
| **Inference Optimization**   | Quantization (FP32→INT8), layer fusion, dynamic batching, KV-cache management (TensorRT)                                      |
| **nvidia-smi**               | Single-node GPU monitoring. Point-in-time. No alerting. Good for debugging.                                                   |
| **DCGM**                     | Enterprise GPU monitoring. SM utilization, ECC errors, thermal throttling, NVLink throughput. Prometheus/Grafana integration. |
| **Container Toolkit**        | Enables `docker run --gpus`. Makes GPUs visible inside containers. Supports Docker, containerd, CRI-O.                        |
| **GPU Operator**             | Kubernetes operator that auto-deploys drivers, device plugin, DCGM, MIG manager. Helm-based.                                  |
| **Slurm**                    | HPC workload manager. GRES for GPUs. Gang scheduling. Common for large-scale training.                                        |
| **MIG**                      | Hardware GPU partitioning. Up to 7 instances. Full isolation. A100/H100/B200+. Free.                                          |
| **vGPU**                     | GPU virtualization for VMs. Time-sliced or MIG-backed. Requires NVIDIA AI Enterprise license.                                 |
| **Checkpointing**            | Periodic save of training state to storage. Fault tolerance for long training runs.                                           |
| **MLOps**                    | DevOps for ML: data/model versioning, CI/CT, model registry, drift monitoring, A/B testing                                    |

---

*Previous: [Module 3 — NVIDIA Technology Stack](Module-3-NVIDIA-Technology-Stack.md) | Back to: [README](README.md)*
