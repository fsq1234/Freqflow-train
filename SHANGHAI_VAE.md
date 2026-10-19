# Shanghai 专用 VAE

按原 Meteo VAE 的网络、损失和训练方法，在上海数据上从头训练。
上海的模型定义、训练、潜变量生成和 Shell 管理均放在独立的 Shanghai 文件中，
不修改或调用 Meteo 的训练及管理入口。
不加载之前的 Meteo 权重。当前 PercpCast E1 仍是像素空间 FM；训练本 VAE
不会自动改变 E1，也不会自动开始潜空间 FM 训练。

## 数据与模型

- 使用原始 `shanghai.h5`：`train/{index}`、`test/{index}`，各组数量由
  `{split}/all_len` 读取。
- 沿用当前上海预处理：原始像素除以 255，再 Resize 到 128×128；验证仍按
  现有协议映射到 test。原始 HDF5 不会被改写。
- 评估沿用上海的 `value_scale=90`、阈值 `[20,30,35,40]`。
  评估比例和输入除数是两个不同参数。
- 每个事件每轮随机读取一帧，沿用 Meteo 的 VAE 采样方法，不把一个事件
  扩展成 25 个优化样本。
- 使用相同 AutoencoderKL：`1×128×128 -> 4×32×32 -> 1×128×128`，
  通道 128/256/512，KL 与自适应 PatchGAN 损失保持原设置。
- 默认 batch=3、250 轮、LR=1e-4、FP32；GAN 启用条件继续使用原来的
  `epoch > disc_start_epoch`，默认 `disc_start_epoch=35`。

## 启动与续训

在原来运行 Meteo VAE 的 Linux 环境中，将本分支代码同步到 FreqFlow，
进入项目目录后运行：

```bash
bash train_manager_shanghai_vae32.sh command  # 只打印参数，不启动训练
bash train_manager_shanghai_vae32.sh start
bash train_manager_shanghai_vae32.sh status
bash train_manager_shanghai_vae32.sh dir
bash train_manager_shanghai_vae32.sh resume
```

默认数据路径为 `/mnt/sdb1/user/WRF/stlP/data/shanghai.h5`。
需要修改时使用上海脚本提供的环境变量：

```bash
FREQFLOW_DATA_PATH=/your/path/shanghai.h5 bash train_manager_shanghai_vae32.sh start
```

上海输出独立保存到：

```text
runs/shanghai-vae32/<timestamp>/
  checkpoints/last.pt
  checkpoints/best_val_loss.pt
  checkpoints/best_csi.pt
  logs/train.log
  vae_config.json
  data_config.json
  run_config.json
  metrics.jsonl
runs/shanghai_vae32_latest_run.txt
```

正式权重仍选择 GAN 激活阶段的验证生成损失最优 `best_val_loss.pt`；
`best_csi.pt` 用于诊断。训练初期尚未激活 GAN 时，不会产生正式的
`best_val_loss.pt`。`--resume` 用于恢复上海 VAE 自身的训练，不能接续
Meteo 权重作为上海实验。

直接前台训练也可以使用：

```bash
python train_vae_shanghai.py \
  --data-path /mnt/sdb1/user/WRF/stlP/data/shanghai.h5 \
  --workdir runs/shanghai-vae32/manual \
  --batch-size 3 --epochs 250 --num-workers 12 --mixed-precision no
```

## 训练后生成潜变量（可选）

先确认重建结果可接受，再选择这一轮上海 VAE 的正式权重：

```bash
export FREQFLOW_VAE_CHECKPOINT=/your/run/checkpoints/best_val_loss.pt
bash generate_shanghai_latents.sh
```

默认新文件为 `data/shanghai_latent_vae32.h5`。每个事件保持 25 帧时，
潜变量形状为 `[25,4,32,32]`；使用 `posterior.mode()`，均值/标准差只由
train 潜变量统计。编码过程分块运行，原始上海文件保留供第一阶段使用。
这份缓存不能作为原始像素 HDF5 直接交给当前 E1；潜空间 FM 的状态通道、
编码/解码及缓存读取接口需要随后单独接入。

## 验证范围

```bash
python -m unittest -v test_shanghai_vae
```

合成 HDF5 检查包括：上海 /255 与 Resize 一致、验证映射一致、Meteo
默认行为保留、跨数据集权重拒绝、潜变量事件/帧顺序与形状、只使用训练集
统计标准化参数、原始文件不被改写。

潜变量文件检查使用一个池化编码器替身，只验证数据链路，不代表真实 VAE
重建质量。本地缺少 Accelerate/Diffusers，因此真实 VAE 前后向、GAN 训练、
服务器续训和真实数据指标仍需在原 Meteo 训练环境中验证。
