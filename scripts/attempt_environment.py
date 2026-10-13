"""Record the portable pinned Conda attempt, including a missing executable."""
import json
import subprocess
from pathlib import Path
command = ["conda", "env", "create", "-f", "envs/reproduction.yml"]
try:
    p = subprocess.run(command, capture_output=True, text=True)
    result = {"command": command, "returncode": p.returncode, "stdout": p.stdout, "stderr": p.stderr}
except OSError as exc:
    result = {"command": command, "error": str(exc), "exception_type": type(exc).__name__}
Path("envs/pinned_attempt.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
