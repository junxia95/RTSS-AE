# Expected results

`source/` contains immutable table and figure data exported from the authors' completed runs.
`source/fig8/` contains the complete physical-testbed server logs, original
HP 9800 power exports, session metadata, and SHA256 manifest used to reproduce
both panels of Figure 8 without access to Jetson hardware.
Private LAN prefixes in the public server logs are replaced with RFC 5737
documentation addresses; all timestamps, metrics, rounds, and power records
remain unchanged, and the public-release hashes are recorded in the manifest.
`raw/tinyimagenet_round_metrics/` contains the per-round accuracy, time, and
communication series used to recompute Table II; `manifest.csv` records each
file's original path and SHA256 digest.
Reproduction scripts write new artifacts under `results/reproduced/` and never modify these files.
