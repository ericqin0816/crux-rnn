"""Freeze once, after all pilot cells verify. Never overwrite an existing freeze."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from motor_rnn.experiment import verify_done
from motor_rnn.provenance import file_hash, implementation_hash, write_json

pilot = json.loads(Path("configs/pilot.json").read_text())
assignment = json.loads(Path(pilot["output_directory"], "blocked_order_assignment.json").read_text())
for r in pilot["run_ids"]:
    for k in pilot["distances"]:
        cfg = {**pilot, "run_id": r, "distance": k, "blocked_order": assignment[str(k)][str(r)]}
        assert verify_done(Path(pilot["output_directory"], "runs", "run_{:03d}_k{}".format(r, k)), cfg)
path = Path("configs/final.json")
assert not path.exists(), "Final configuration already frozen; never overwrite"
config = json.loads(Path("configs/final_candidate.json").read_text())
config["implementation_sha256"] = implementation_hash()
write_json(path, config)
path.with_suffix(".sha256").write_text(file_hash(path) + "\n")
print(file_hash(path))
