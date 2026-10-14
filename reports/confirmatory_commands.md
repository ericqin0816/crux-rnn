# CruxRNN: approved final execution and verification

The user approved the pilot and explicitly authorized pushes, final execution, tag and release. The frozen configuration hash was verified before any fresh data were generated.

1. Verified clean main at fa3b7c34496274569944b85c74d2c6c4cd443fe4, correct origin, ignored author checkout, and no tracked credential patterns/checkpoints/temporary or >10MB files.
2. Ran git push -u origin main; git ls-remote origin refs/heads/main returned the exact expected pre-final SHA.
3. Ran the exact interpreter's unittest suite (17 passed) and scripts/system_info.py; preserved new environment separately from historical evidence.
4. Recorded user approval and descriptive SD/group-table/300-dpi additions in producing commit 28a2b07bb40afcf7af9b907612e01ce722902bcc. Pushed it. Frozen numerical modules/config were unchanged.
5. Copied configs/final_proposed_blocked_orders.json byte-for-byte into the fresh output directory. Ran scripts/run_experiment.py --config configs/final_proposed.json once, without --resume, --force or --limit. All 120 cells completed.
6. Ran scripts/analyze.py --config configs/final_proposed.json; its contemporaneous execution log records successful analysis and figure generation.
7. Ran scripts/verify_design.py --config configs/final_proposed.json; all 120 cells, 1560 saved seeds, data/completion hashes and bootstrap row resampling verified.
8. Ran scripts/publish_confirmatory.py and scripts/create_walkthrough.py. Inspected all five PNGs. All eight notebook cells executed. Reporting helpers do not train models.
9. Ran unittest after final training (17 passed). Verified curated hashes and decompressed histories against original CSV bytes; independently recalculated primary means, medians, ddof=1 SDs and bootstrap percentiles.
10. Ran scripts/archive_confirmatory.py; checked all archive members' CRCs and recorded the complete archive SHA-256. Only the new final directory is packaged.

Final results and an explicit SHA-recording follow-up are committed/pushed after these checks. The requested tag and browser release follow verified remote delivery. Numerical outcome, elapsed time and producing-SHA facts are in RUN_REPORT and the execution record; browser delivery status is separate.
