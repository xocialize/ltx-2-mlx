# scripts/ — internal development tooling

⚠️ **These are not user-facing, and most will not run as-is on your machine.** They are the
conversion, golden-dump and parity-gate scripts used to develop and verify the port, and many
hardcode absolute paths to a specific development volume (`/Volumes/Satechi/...`) and to a local
checkout of the PyTorch reference.

They are kept in the repository because they are the *provenance* of the numbers quoted in
[`docs/claims/`](../docs/claims/) and in the README — the parity gates, the quantization recipes,
and the measurement harnesses. Reading them tells you exactly how a figure was produced.

To actually use one, expect to edit the paths at the top. For normal generation you want the CLI
(`ltx-2-mlx --help`), not anything in here.
