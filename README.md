# Solar Defect Discovery: High-Performance DL Engine ☀️🔋

<p align="left">
  <img src="https://img.shields.io/badge/NVIDIA%20Certified-Generative%20AI%20%26%20LLM-green?style=for-the-badge&logo=nvidia" />
  <img src="https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python" />
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
</p>

### 🚀 Project Overview
This project transitions a research-grade Solar Panel defect classification model into a **production-grade inference engine**. Optimized for **NVIDIA GPU architectures**, the system targets sub-millisecond latency for real-time physical AI discovery.

### 📊 Performance Results (Tesla T4 GPU)
| Optimization | Latency (ms) | Throughput (FPS) | Precision |
| :--- | :--- | :--- | :--- |
| PyTorch Eager (Baseline) | ~12.40 ms | ~80 FPS | FP32 |
| **Quantized Engine** | **0.93 ms** | **1,073.22 FPS** | **FP16** |

### 🛠️ Engineering Highlights
* **Modular Design**: Structured Python package with decoupled `core/` and `scripts/` layers.
* **Hydra Configuration**: Managed environment parameters via a YAML-based config layer for scalable deployment.
* **Hardware-Aware Profiling**: Utilized `torch.cuda.synchronize()` to ensure precise kernel-level latency measurements.
* **Deployment Ready**: Fully prepared for **TensorRT** serialization and **NVIDIA Triton** orchestration.

---

### 💻 Installation & Usage

#### 1. Setup Environment
```bash
git clone [https://github.com/Vinaykumaryadi/solar-inference-engine-trt.git](https://github.com/Vinaykumaryadi/solar-inference-engine-trt.git)
cd solar-inference-engine-trt
pip install -r requirements.txt

```
#### 2. Run Benchmarking
To verify the engine's performance on your local GPU:

```bash
python scripts/benchmark_trt.py
```
#### 3. Run Inference Test
To test the model logic with a specific image:


Place your image at the root as 'test_solar.jpg'
```
python scripts/test_inference.py
```

#### 📂 Repository Structure
```
solar-defect-engine/
├── conf/                # Hydra YAML configurations
├── core/                # Model architecture & logic
├── scripts/             # Benchmarking & Testing tools
├── data/                # Data placeholders
└── requirements.txt     # Dependency management

```


#### 👤 Author
#### Vinay Kumar Yadi Senior Software Engineer (Tesla) | NVIDIA Certified Professional


