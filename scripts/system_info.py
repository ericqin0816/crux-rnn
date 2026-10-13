"""Capture execution environment. Run this with the experiment interpreter."""
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
import torch


def run(command):
    try:
        p = subprocess.run(command, capture_output=True, text=True)
        return {"returncode": p.returncode, "stdout": p.stdout, "stderr": p.stderr}
    except OSError as exc:
        return {"error": str(exc)}


if __name__ == "__main__":
    target = Path("outputs/environment")
    target.mkdir(parents=True, exist_ok=True)
    freeze = run([sys.executable, "-m", "pip", "freeze"])
    (target / "pip_freeze.txt").write_text(freeze.get("stdout", ""), encoding="utf-8")
    ps = "Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors | ConvertTo-Json"
    ram = "Get-CimInstance Win32_OperatingSystem | Select-Object Caption,FreePhysicalMemory,TotalVisibleMemorySize | ConvertTo-Json"
    info = {"python": sys.version, "executable": sys.executable, "os": platform.platform(),
            "cpu": run(["powershell", "-NoProfile", "-Command", ps]),
            "memory": run(["powershell", "-NoProfile", "-Command", ram]),
            "logical_cpu_count": os.cpu_count(), "torch": torch.__version__,
            "torch_device": "cpu", "torch_default_threads": torch.get_num_threads(),
            "experiment_threads": 1, "torch_interop_threads": torch.get_num_interop_threads(),
            "cuda_available": torch.cuda.is_available(), "torch_cuda_version": torch.version.cuda,
            "mps_available": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
            "conda_environment": run(["conda", "env", "export"]),
            "conda_prefix": os.environ.get("CONDA_PREFIX"), "torch_build": torch.__config__.show(),
            "accelerator_policy": "CPU only; no accelerator used"}
    if torch.cuda.is_available():
        info["cuda_devices"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    (target / "system_info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(info, indent=2))
