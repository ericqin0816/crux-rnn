"""Measure CPU thread overhead for this tiny model before choosing acceleration."""
import json
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch
from motor_rnn.model import MotorRNN
from motor_rnn.provenance import write_json
from motor_rnn.training import optimizer_for, step

rows = []
torch.set_num_interop_threads(1)
for threads in (1, 2, 4):
    torch.set_num_threads(threads)
    torch.manual_seed(1873)
    model = MotorRNN()
    opt = optimizer_for(model)
    x = torch.randn(20, 1, 7) + 2.5
    y = torch.randint(4, (20, 7))
    for _ in range(10):
        step(model, opt, x, y)
    begin = time.perf_counter()
    for _ in range(100):
        step(model, opt, x, y)
    rows.append({"device": "cpu", "threads": threads, "updates": 100, "seconds": time.perf_counter() - begin})
write_json("outputs/environment/benchmark.json", {"measurements": rows, "cuda_available": torch.cuda.is_available(),
                                                    "extension_device": "cpu", "acceleration_used": False})
print(json.dumps(rows, indent=2))
