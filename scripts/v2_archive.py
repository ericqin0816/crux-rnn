"""Package the completed v2 raw study only, with content hashes and CRC audit."""

import json
import zipfile
from pathlib import Path

from cruxrnn.config import load_config
from cruxrnn.reproducibility import file_hash, utc_now, write_json


def main():
    config = load_config("configs/v2/main.json")
    root = Path(config.output_directory)
    verification = json.loads((root / "analysis/verification.json").read_text())
    if verification["completed_cells"] != 1260 or verification["missing_cells"]:
        raise RuntimeError("Only a fully accounted-for study can be packaged as complete")
    target = Path("artifacts/cruxrnn-v2-research-data.zip")
    if target.exists():
        raise FileExistsError("Retain the existing raw archive; do not overwrite it")
    files = sorted(path for path in root.rglob("*") if path.is_file())
    files += [
        Path("configs/v2/main.json"),
        Path("configs/v2/main.sha256"),
        Path(config.plan),
        Path("provenance/v2_environment/system_info.json"),
    ]
    manifest = {
        "namespace": config.namespace,
        "created_at_utc": utc_now(),
        "producing_git_commits": verification["producing_git_commits"],
        "configuration_file_sha256": file_hash("configs/v2/main.json"),
        "main_models": 2760,
        "D_adapted_clones": 2760,
        "independent_main_families": 700,
        "v1_pilot_smoke_or_demo_data_included": False,
        "files": {},
    }
    with zipfile.ZipFile(
        str(target), "x", zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True
    ) as archive:
        for path in files:
            manifest["files"][path.as_posix()] = file_hash(path)
            archive.write(
                str(path),
                path.as_posix(),
                compress_type=zipfile.ZIP_STORED if path.suffix == ".npz" else zipfile.ZIP_DEFLATED,
            )
        archive.writestr(
            "archive_manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        archive.writestr(
            "README.txt",
            "CruxRNN v2 raw scientific study. No v1, smoke, pilot or demo data.\n"
            "Extract at the root of the v2 review-branch checkout.\n"
            "The archive populates ignored outputs/v2/main; compact results remain in Git.\n"
            "Run cruxrnn verify --config configs/v2/main.json with the recorded environment.\n"
            "The internal manifest contains each payload SHA-256 and producing Git identity.\n"
            "2760 main models and 2760 D-adapted clones are not 5520 independent subjects.\n"
            "Final v2 release awaits owner review; this local artifact is not a published release.\n",
        )
    with zipfile.ZipFile(str(target)) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("Archive CRC verification failed")
        members = len(archive.namelist())
    receipt = {
        "archive": target.as_posix(),
        "bytes": target.stat().st_size,
        "sha256": file_hash(target),
        "members": members,
        "all_members_crc_verified": True,
        "payload_files_have_internal_sha256": True,
        "published_release_created": False,
        "namespace": config.namespace,
        "main_models": 2760,
        "D_adapted_clones": 2760,
        "contains_historical_or_diagnostic_data": False,
    }
    write_json("artifacts/v2_archive_receipt.json", receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
