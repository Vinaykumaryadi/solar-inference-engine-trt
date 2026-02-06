# Solar Defect Discovery: High-Performance DL Engine ☀️🔋

[![NVIDIA Certified](https://img.shields.io/badge/NVIDIA-Certified_GenAI_%26_LLM-green.svg)](https://www.nvidia.com/)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-red.svg)](https://pytorch.org/)

### 🚀 Project Overview
This project transitions a research-grade Solar Panel defect classification model into a **production-ready inference engine**. Optimized for **NVIDIA GPU architectures**, the system focuses on maximizing throughput and minimizing latency—key requirements for **Physical AI** and autonomous discovery environments.



### 🛠️ Senior DL Engineering Highlights
* **Modular Architecture**: Refactored monolithic scripts into a structured Python package (`core/` module) to support maintainability and scalability.
* **Decoupled Configuration**: Utilized **Hydra** for a dedicated `conf/` layer, enabling seamless environment-specific tuning across different GPU clusters without code changes.
* **Optimization**: Implemented **FP16 Mixed Precision** and **Adaptive Pooling** to maximize GPU kernel occupancy and reduce memory footprint.
* **Deployment Pipeline**: Supports **ONNX** export with dynamic batching, prepared for high-scale deployment via **NVIDIA Triton Inference Server** or **TensorRT**.
* **Hardware Profiling**: Benchmarked using **CUDA-synchronization** (`torch.cuda.synchronize()`) to capture accurate, professional-level hardware latency.

### 📊 Performance Results (Tesla T4 GPU)
*Captured using Python-based profiling tools on NVIDIA hardware[cite: 1, 2].*

| Optimization | Latency (ms) | Throughput (FPS) | Precision |
| :--- | :--- | :--- | :--- |
| PyTorch Eager | *~12.40 ms** | *~80 FPS* | FP32 |
| **Quantized Engine** | **0.93 ms** | **1,073.22 FPS** | **FP16** |

### 📂 Repository Structure
```text
solar-defect-engine/
├── conf/                # Hydra Configuration (Twelve-Factor App standards)
├── core/                # Production Model Logic (PyTorch)
├── scripts/             # Benchmarking & Export Tools
├── data/                # Dataset placeholder (Local/Cloud storage)
└── solar_model.onnx     # Deployment-ready ONNX weights
