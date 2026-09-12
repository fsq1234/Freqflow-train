# FreqFlow nowcasting contract

`alphapre_compat/` directly contains AlphaPre's original data and metric
sources: `dataset_sevir.py`, `dataset_cikm.py`, `dataset_meteonet.py`,
`dataset_shanghai.py`, `get_datasets.py`, `metrics_valid.py`, and `metrics.py`.
`train_nowcast_alphapre.py` imports those local copies, so SEVIR VIL reading,
date splits, `seq_len=25`, stride `13`, `/255` scaling, `5 -> 20` frame split,
thresholds `(16, 74, 133, 160, 181, 219)`, CSI checkpoint selection, and full
test reporting follow AlphaPre's original implementation without requiring an
external AlphaPre checkout.

The supplied training entrypoint keeps AlphaPre's `5 -> 20` contract and
supports `--dataset sevir` (default) and `--dataset meteo`.  SEVIR uses its
original internally batched reader. MeteoNet uses AlphaPre's original ordinary
PyTorch `DataLoader` path and takes the single `meteo_radar.h5` file as
`--data-path`. The CIKM and Shanghai readers are included for later
dataset-specific launchers; their input/output frame contract must be set
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

For AlphaPre-format MeteoNet data:

```bash
accelerate launch train_nowcast_alphapre.py \
  --dataset meteo \
  --data-path /mnt/sdb1/user/WRF/stlP/data/meteo_radar.h5 \
  --batch-size 1 \
  --mixed-precision fp16 \
  --valid
```

AlphaPre's original `Meteo` reader maps `val` to the HDF5 `test` group because
that file has only `train` and `test` splits. Thus, with `--valid`, validation
and the final test report use the same held-out HDF5 group; this launcher keeps
that source behavior rather than silently inventing a new split.

## Background launchers

`train_manager.sh` and `train_manager_meteonet.sh` provide the same
`start`, `resume`, `stop`, `status`, and `dir` workflow as the existing
PercpCast manager scripts. They resolve the FreqFlow project directory from
their own location rather than embedding a server-specific checkout path.

```bash
bash train_manager_meteonet.sh start
bash train_manager_meteonet.sh status
bash train_manager_meteonet.sh resume

bash train_manager.sh start
```

The data paths, visible GPU, per-process batch size, epoch count, precision,
and validation mode are at the top of each dataset launcher. The defaults are
the current server paths, GPU `0`, batch size `1`, `fp16`, and 100 epochs.
`FREQFLOW_FINAL_TEST_CHECKPOINT=last` keeps the original behavior of testing
the final EMA; set it to `best` to load `best.pt`'s EMA before the automatic
full test.

The flow state is the normalized future VIL sequence `[B,20,H,W]`; the history
is `[B,5,H,W]`.  It uses FlowCast's conditional-flow-matching direction
(`noise -> future`) while retaining FreqFlow's low/high-frequency velocity
supervision.  Before AlphaPre's unmodified metric code is called, both tensors
are restored to its native `[B,20,1,H,W]` format.  Each checkpoint records the
resolved model/task configuration as `run_config`, so a server-side run can be
traced back to this exact conditioning contract.
