# NCA-AIIO Practice Exam

> **Instructions:** Read each question carefully and select the best answer before revealing the solution. Click on the **Show Answer** toggle below each question to reveal the correct answer and detailed explanations.

---

## Question 1

**Domain:** AI Infrastructure

Your organization is evaluating DGX SuperPOD configurations for large-scale LLM training workloads requiring 1 ExaFLOP of compute performance at FP8 precision. Which DGX SuperPOD Scalable Unit (SU) configuration best meets this performance requirement?

- A) DGX A100 SuperPOD SU with 20 DGX A100 systems providing 1 ExaFLOP FP8 performance per SU
- B) DGX H100 SuperPOD SU with 32 DGX H100 systems providing 1 ExaFLOP FP8 performance per SU
- C) DGX GH200 SuperPOD SU with 256 GH200 Superchips providing 1 ExaFLOP FP8 performance per SU
- D) DGX H100 SuperPOD SU with 64 DGX H100 systems providing 2 ExaFLOPs FP8 performance per SU

<details>
<summary>Show Answer</summary>

### Correct Answer: **B**

---

**B) DGX H100 SuperPOD SU with 32 DGX H100 systems providing 1 ExaFLOP FP8 performance per SU** ✅

A DGX H100 SuperPOD Scalable Unit contains 32 DGX H100 systems (256 H100 GPUs total), delivering exactly 1 ExaFLOP of FP8 Tensor Core performance per SU. This is the standard configuration for ExaFLOP-scale workloads.

---

**Why the other answers are incorrect:**

**A) DGX A100 SuperPOD SU with 20 DGX A100 systems** ❌
DGX A100 systems lack 4th generation Tensor Cores and do not support FP8 precision natively, only FP16/BF16. A100-based SUs cannot deliver 1 ExaFLOP at FP8 precision. The Hopper architecture (H100) is required for FP8 compute.

**C) DGX GH200 SuperPOD SU with 256 GH200 Superchips** ❌
DGX GH200 configurations deliver significantly higher performance than 1 ExaFLOP per SU due to their extreme-scale architecture. The standard 1 ExaFLOP FP8 performance benchmark specifically refers to DGX H100 SuperPOD SU configurations, not GH200.

**D) DGX H100 SuperPOD SU with 64 DGX H100 systems** ❌
While 64 DGX H100 systems would deliver approximately 2 ExaFLOPs at FP8, this exceeds the standard SU definition. A single DGX H100 SuperPOD SU is defined as 32 systems delivering 1 ExaFLOP, not 64 systems.

---

### Key Concept

A DGX H100 SuperPOD Scalable Unit (SU) is specifically architected to deliver 1 ExaFLOP of FP8 Tensor Core performance, comprising 32 DGX H100 systems with 256 H100 GPUs total. This configuration leverages Hopper's 4th generation Tensor Cores with native FP8 support and the Transformer Engine, making it the standard building block for ExaFLOP-scale AI infrastructure deployments.

</details>

---

*More questions coming soon...*
