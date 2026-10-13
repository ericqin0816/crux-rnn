# CruxRNN: commands and actions during pilot review

Interpreter: `envs/tools/python3810/tools/python.exe`, the exact local Python 3.8.10 runtime. No final-experiment runner command was issued during this review. Existing simulations were reverified, not retrained.

```powershell
git branch archive/pre-cruxrnn-execution 2673c9ca11ca9fc701a1d13129f0cb371385ea70
git branch -m main
& 'envs/tools/python3810/tools/python.exe' -m unittest discover -s tests -v
& 'envs/tools/python3810/tools/python.exe' scripts/verify_design.py --config configs/source_exact.json
& 'envs/tools/python3810/tools/python.exe' scripts/verify_design.py --config configs/smoke.json
& 'envs/tools/python3810/tools/python.exe' scripts/verify_design.py --config configs/pilot.json
& 'envs/tools/python3810/tools/python.exe' scripts/prepare_pilot_review.py
& 'envs/tools/python3810/tools/python.exe' scripts/create_walkthrough.py
git remote add origin https://github.com/ericqin0816/crux-rnn.git
git ls-remote origin
git diff --check
git commit -m "Name CruxRNN and gate additional final execution pending review"
git commit -m "Publish verified CruxRNN pilot artifacts and unexecuted confirmation proposal"
& 'envs/tools/python3810/tools/python.exe' scripts/write_pilot_report.py
```

Tests were rerun after the approval/provenance changes and still passed 17/17. Curated manifests, historical snapshot bytes, proposal-only differences, balanced orders, and unchanged frozen implementation were checked with read-only Python assertions. Original execution commands remain in `reports/commands.md` and historical documents.

Browser actions: the user requested use of their logged-in browser. GitHub's creation form visibly confirmed owner `ericqin0816`; `crux-rnn` was available, and an empty public repository was created with the exact requested description. The owner/URL/public/empty state was verified. Screenshot: `artifacts/cruxrnn_repo_created.png`. The mismatched connector was used only for initial read-only identity/availability checks, never for writes. No repository files were uploaded or pushed. No final tag/release was created.
