import hashlib
import json
import subprocess
from pathlib import Path
root = Path("external/author_repo")
commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
assert commit == "31f58bb6de90d1fffcb752c0b97afb7df51479a7"
files = ["Code01_Interleaved_vs_Repetitive.py", "Code02_Plot.py", "utils/DataGen.py", "utils/NetworkModels.py", "utils/TorchFunctions.py", "environment.yml"]
output = {"source_commit": commit, "repository": "https://github.com/YJ-0000/RNN_during_RP_and_IP",
          "root_LICENSE_found": (root / "LICENSE").exists(),
          "file_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in files}}
target = Path("outputs/provenance")
target.mkdir(parents=True, exist_ok=True)
(target / "source_files.json").write_text(json.dumps(output, indent=2) + "\n")
(target / "source_commit.txt").write_text(commit + "\n")
print(json.dumps(output, indent=2))
