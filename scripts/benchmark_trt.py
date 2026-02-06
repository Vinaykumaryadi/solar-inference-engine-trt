import torch
import time
import hydra
from omegaconf import DictConfig
import sys

# Ensure core is importable
sys.path.append('/content/solar-defect-engine')
from core.model import SolarCNN

@hydra.main(config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    device = torch.device(cfg.hardware.device if torch.cuda.is_available() else "cpu")
    model = SolarCNN(num_classes=cfg.model.num_classes).to(device).eval()
    
    # Example of using the config for optimization
    if cfg.optimization.precision == "fp16" and cfg.hardware.device == "cuda":
        model = model.half()
        print("Using FP16 Precision for Inference")

    dummy_input = torch.randn(1, *cfg.model.input_size).to(device)
    if cfg.optimization.precision == "fp16":
        dummy_input = dummy_input.half()

    # Warmup
    for _ in range(10):
        _ = model(dummy_input)

    # Performance Benchmarking with Synchronization
    if cfg.hardware.sync_gpu:
        torch.cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(100):
        _ = model(dummy_input)
    
    if cfg.hardware.sync_gpu:
        torch.cuda.synchronize()
    end = time.perf_counter()

    latency = (end - start) / 100 * 1000
    print(f"--- {cfg.model.name} Benchmark ---")
    print(f"Latency: {latency:.4f} ms")
    print(f"Throughput: {1000/latency:.2f} FPS")

if __name__ == "__main__":
    main()
