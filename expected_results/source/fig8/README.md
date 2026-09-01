# Figure 8 source records

This directory contains the immutable records used to reproduce Figure 8.
Each method has one UTF-8 server log and three raw HP 9800 active-power
streams per measurement session. The meter exports are preserved in their
original GBK encoding and sample active power at approximately 1 Hz.

For public release, private testbed addresses in the three server logs were
replaced with the documentation-only `192.0.2.0/24` prefix. No timestamps,
metrics, round records, or power measurements were changed. `manifest.csv`
contains the hashes of these public-release files.

RTDFL rounds 0-64 were collected on 2026-05-25. The run resumed with rounds
65-99 on 2026-05-26, so RTDFL has two power sessions. DFL and MMDFL each have
one session. `metadata.json` records these dates because the original meter
exports contain time-of-day but no calendar date.

The plot uses average test accuracy (`avg_acc`) from each completed round.
Cumulative active runtime is the sum of completed-round durations in the
server log; it is not end-to-end elapsed wall-clock time and excludes the
overnight RTDFL interruption. Round energy is obtained by integrating each
covered power stream over the corresponding round timestamps and summing the
three meters. The remaining-energy axis is a nominal accounting budget: it
starts at 150 kJ per device, or 1500 kJ across ten devices, and subtracts the
integrated active energy. It is not a direct battery state-of-charge reading.

From the repository root, run:

```bash
python scripts/reproduce_fig8.py
```

The command requires all three methods to contain rounds 0-99 and requires all
three power streams to cover at least 95 percent of every completed round. It
first verifies all immutable inputs against `manifest.csv`, then writes the two
Figure 8 panels and a 300-row audit table under `results/reproduced/`.

`manifest.csv` records the size and SHA256 digest of every immutable server log
and meter export in this directory.
