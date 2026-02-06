import torch
import time
import hydra
from omegaconf import DictConfig
import sys
sys.path.append('/content/solar-defect-engine')
from core.model import SolarCNN

@hydra.main(config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    device = torch.device(cfg.hardware.device if torch.cuda.is_available() else "cpu")
    model = SolarCNN(num_classes=cfg.model.num_classes).to(device).eval()
    if cfg.optimization.precision == "fp16": model = model.half()
    
    dummy_input = torch.randn(1, *cfg.model.input_size).to(device)
    if cfg.optimization.precision == "fp16": dummy_input = dummy_input.half()

    for _ in range(20): _ = model(dummy_input) # Warmup
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(200): _ = model(dummy_input)
    torch.cuda.synchronize()
    latency = (time.perf_counter() - start) / 200 * 1000
    print(f"Latency: {latency:.4f} ms | Throughput: {1000/latency:.2f} FPS")

if __name__ == "__main__":
    main()
