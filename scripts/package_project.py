"""CruxRNN: bundle own source and generated outputs, excluding reference/runtime code."""
import json
import sys
import zipfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from motor_rnn.provenance import file_hash, write_json

root = Path.cwd()
files = [root / name for name in (".gitignore", ".gitattributes", "README.md", "RUN_REPORT.md", "pyproject.toml")]
for name in ("src", "tests", "scripts", "reports", "configs", "notebooks", "envs", "outputs"):
    for path in (root / name).rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if relative.as_posix() == "reports/package_execution_log.txt":
            continue  # This log is being written by the current package command.
        if "__pycache__" in relative.parts or (len(relative.parts) >= 2 and relative.parts[:2] == ("envs", "tools")):
            continue
        if path.suffix in (".pyc", ".tmp"):
            continue
        files.append(path)
manifest = {"final_config_sha256": file_hash("configs/final.json"), "source_commit": "31f58bb6de90d1fffcb752c0b97afb7df51479a7",
            "excludes": ["external/author_repo", "envs/tools", ".venv", ".git", "__pycache__"],
            "files": {p.relative_to(root).as_posix(): file_hash(p) for p in sorted(files)}}
artifacts = root / "artifacts"
artifacts.mkdir(exist_ok=True)
destination = artifacts / "crux-rnn-project.zip"
with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=3) as archive:
    archive.writestr("PACKAGE_MANIFEST.json", json.dumps(manifest, indent=2) + "\n")
    for path in sorted(files):
        archive.write(path, path.relative_to(root).as_posix())
with zipfile.ZipFile(destination) as archive:
    assert archive.testzip() is None, "Archive CRC verification failed"
    assert not any(name.startswith(("external/", "envs/tools/", ".venv/", ".git/")) for name in archive.namelist())
write_json(artifacts / "delivery_manifest.json", {"archive": destination.name, "sha256": file_hash(destination),
                                               "bytes": destination.stat().st_size, "included_files": len(files),
                                               "archive_crc_verified": True, "final_config_sha256": manifest["final_config_sha256"]})
print(json.dumps(json.loads((artifacts / "delivery_manifest.json").read_text()), indent=2))
