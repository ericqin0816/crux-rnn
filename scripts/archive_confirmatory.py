"""Package only the new confirmation's full raw data, with verified hashes."""
import json
import sys
import zipfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from motor_rnn.provenance import file_hash, write_json

cfg_path = Path("configs/final_proposed.json")
cfg = json.loads(cfg_path.read_text())
root = Path(cfg["output_directory"])
design = json.loads((root / "analysis/design_verification.json").read_text())
assert design["verified_cells"] == 120 and design["missing_cells"] == 0
files = sorted(p for p in root.rglob("*") if p.is_file())
manifest = {"project": "CruxRNN", "namespace": cfg["namespace"],
            "config_sha256": file_hash(cfg_path), "paired_cells": 120, "main_models": 240,
            "raw_directory": root.as_posix(), "earlier_exploratory_data_included": False,
            "files": {p.as_posix(): file_hash(p) for p in files}}
target = Path("artifacts/cruxrnn-confirmatory-data.zip")
target.parent.mkdir(exist_ok=True)
with zipfile.ZipFile(str(target), "w", zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as archive:
    for path in files:
        archive.write(str(path), path.as_posix())
    archive.writestr("archive_manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    archive.writestr("README.txt", "CruxRNN fresh confirmation only. Extract at the root of the release checkout.\n"
                     "Raw data populate the ignored outputs/cruxrnn_final_proposed_v1 directory.\n"
                     "Verify individual SHA-256 hashes with archive_manifest.json.\n"
                     "The release's frozen config is configs/final_proposed.json.\n"
                     "Use scripts/verify_design.py --config configs/final_proposed.json for the full audit.\n"
                     "Do not merge earlier exploratory or pilot data into final analysis.\n")
with zipfile.ZipFile(str(target)) as archive:
    assert archive.testzip() is None
    assert len(archive.namelist()) == len(files) + 2
write_json("artifacts/confirmatory_archive.json", {"archive": target.as_posix(), "sha256": file_hash(target),
                                                 "bytes": target.stat().st_size, "file_count": len(files) + 2,
                                                 "archive_crc_verified": True, "config_sha256": file_hash(cfg_path),
                                                 "paired_cells": 120, "main_models": 240,
                                                 "earlier_exploratory_data_included": False})
print(json.dumps(json.loads(Path("artifacts/confirmatory_archive.json").read_text()), indent=2))
