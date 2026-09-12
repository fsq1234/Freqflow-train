# FreqFlow nowcasting contract

`alphapre_compat/` directly contains AlphaPre's original data and metric
sources: `dataset_sevir.py`, `dataset_cikm.py`, `dataset_meteonet.py`,
`dataset_shanghai.py`, `get_datasets.py`, `metrics_valid.py`, and `metrics.py`.
`train_nowcast_alphapre.py` imports those local copies, so SEVIR VIL reading,
date splits, `seq_len=25`, stride `13`, `/255` scaling, `5 -> 20` frame split,
thresholds `(16, 74, 133, 160, 181, 219)`, CSI checkpoint selection, and full
test reporting follow AlphaPre's original implementation without requiring an
external AlphaPre checkout.

The supplied training entrypoint is intentionally fixed to AlphaPre's SEVIR
`5 -> 20` contract. The CIKM, MeteoNet and Shanghai readers are included for
later dataset-specific launchers; their input/output frame contract must be set
before training them with FreqFlow.

`configs/freqflow_nowcast_alphapre.py` is the active model configuration. It
keeps the original FreqFlow backbone capacity: Transformer `dim=960`,
`hidden_dim=1920`, `39` blocks and `16` heads; ConvNeXt `dim=480`,
`hidden_dim=960` and `20` blocks. The unavoidable task-interface changes are
20 output-frame channels, five history-frame condition channels, and spatial
position sizes for 128px radar frames.  The historical sequence is encoded once
by `Conv3x3(5,64) -> SiLU -> Conv3x3(64,64) -> SiLU`; separate `1x1`
projections then add its features to the frequency input `[low(x_t), high(x_t)]`
and to the spatial input `x_t`.  The original frequency-stage skip features are
still added inside the spatial stage, so the two-stage FreqFlow backbone is not
replaced or shortened.

Run from `FreqFlow` with its required dependencies installed and a SEVIR root containing
`CATALOG.csv` and `data/`:

```powershell
accelerate launch train_nowcast_alphapre.py --data-path D:\path\to\sevir2 --valid
```

The flow state is the normalized future VIL sequence `[B,20,H,W]`; the history
is `[B,5,H,W]`.  It uses FlowCast's conditional-flow-matching direction
(`noise -> future`) while retaining FreqFlow's low/high-frequency velocity
supervision.  Before AlphaPre's unmodified metric code is called, both tensors
are restored to its native `[B,20,1,H,W]` format.  Each checkpoint records the
resolved model/task configuration as `run_config`, so a server-side run can be
traced back to this exact conditioning contract.
