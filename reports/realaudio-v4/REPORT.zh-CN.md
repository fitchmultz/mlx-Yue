# RealAudio Tokenizer v4 MLX 移植与端到端集成报告

- 实施模型：gemini-3.8-flash
- 验收要求：实施完成后切回 gpt-6-astra 进行独立审查
- 日期：2026-09-15
- 项目路径：`/Users/vanch/yue2-mlx`

---

## 一、移植目标与交付清单

成功将 Hugging Face [Mothersuperior/yue2-mothersuperior-realaudio-tokenizer-v4](https://huggingface.co/Mothersuperior/yue2-mothersuperior-realaudio-tokenizer-v4) 完整移植为 Apple Silicon 原生 MLX 架构：

1. **模型转换与权重校验** (`src/lyra/realaudio/convert.py`)：
   - 上游固定 Revision：`f2278a2e005dc4ecc421c53a0929f62b3aeb2280`
   - `tokenizer_head_joint_v4.pt` (SHA256: `d23c4f757a05f031134b8471ec84245ec2338966516e1a9e26a17ff300a5f87e`) $\to$ 转换为 MLX 原生 FP32 safetensors (`tokenizer_head.safetensors`, 103 个张量)。
   - `nar_lora_joint_v4.pt` (SHA256: `df175dbf9405a8e15b2c3f8dbdcc97303575f763787f227b03029020e28102fe`) $\to$ 转换为 MLX 原生 BF16 safetensors (`nar_lora.safetensors`, 396 个张量，含 392 个 LoRA A/B 矩阵及完整的 `vae2llm` / `llm2vae` IO 替换权重)。
   - 生成完整来源清单与校验文件：`models/realaudio-v4/converted/manifest.json`。

2. **原生 MLX 特征提取与语义编码** (`src/lyra/mert.py`, `src/lyra/realaudio/tokenizer.py`)：
   - 扩展 MLX MERT2 支持 `target_layer=20` 精准隐藏状态抽取。
   - 实现了 24kHz 单声道重采样、30s 窗口特征提取、25Hz 高精度时间轴重采样插值、整曲实例归一化 (`instance_norm`)。
   - 原生 MLX 实现 8 层 Transformer Tokenizer Head，支持 512 帧窗口拼接与两端边界裁切重叠拼接，映射到 32,768 个 YuE2 语义代码。

3. **内存安全 NAR LoRA 适配器** (`src/lyra/realaudio/adapter.py`)：
   - 支持动态 `apply(nar_model, scale=1.0)` 与 `revert(nar_model)`。
   - 不修改磁盘基础权重，内存中自动备份初始权重，卸载/还原时实现 100% 绝对精确复原（位级别误差 0.000000e+00）。
   - 覆盖全部 28 层 7 个核心线性投影层及 `vae2llm` / `llm2vae`。

4. **端到端流水线与 CLI 命令** (`src/lyra/realaudio/pipeline.py`, `src/lyra/music_tools/realaudio.py`, `src/lyra/cli.py`)：
   - `mlx-yue realaudio tokenize <audio> --output <dir>`：音频语义编码并导出 `tokens.npy` 与元数据。
   - `mlx-yue realaudio reconstruct <audio> --output <dir>`：真实音频语义重建（NAR + VAE）。
   - `mlx-yue realaudio compare <audio> --output <dir>`：同输入、同噪声自动化对比（Stock 32步 vs Adapted 32步 vs Adapted 8步），输出性能指标与网页试听播放器。

5. **技能扩展与本地同步** (`skills/yue2-music/`)：
   - `SKILL.md` 扩展新增 RealAudio v4 真实录音 token 化与 NAR 适配路径。
   - 新增技能调用脚本 `scripts/realaudio_tool.py`。
   - `manifest.json` 升级至版本 0.3.0，更新全功能能力矩阵 `references/capabilities.md`。
   - 完整双向同步至本地 canonical 目录 `/Users/vanch/.skills-manager/skills/yue2-music` 并验证软链接生效。

---

## 二、数值对齐与正确性验证

| 验证项 | 判定标准 | 实测数据 / 证据 | 结论 |
| :--- | :--- | :--- | :--- |
| **Tokenizer Head 转换对齐** | MLX vs PyTorch 前向 logits & Argmax | Max diff = 2.86e-5, Token Match Rate = **100.00%** | **PASS** |
| **25Hz 线性插值精度** | 对比 PyTorch `F.interpolate(mode='linear')` | Mean diff = 1.05e-7, Max diff = 3.72e-5 | **PASS** |
| **NAR LoRA 增量注入** | 校验 196 个线性层与 IO 替换 | 权重实际发生 delta 变化 (diff > 0.01) | **PASS** |
| **NAR LoRA 还原一致性** | 卸载后与基座权重位级别完全一致 | Bit-level diff = **0.000000e+00** | **PASS** |
| **单元测试回归套件** | 全部通过 | 119 passed, 1 skipped, 28 subtests passed | **PASS** |

---

## 三、真实音频基准测试（M3 Max 128GB）

在同一物理机环境、使用相同随机种子（固定 PCG64 初始噪声）进行了端到端对比评测：

### 1. 10.0 秒音频测试 (`reports/realaudio-v4/benchmark_10s`)
- 编码耗时：**0.23 s**（处理速度 43.7× 实时），生成 250 个语义 token (25 Hz)
- 音频波形：无爆音、无静音截断，48kHz FLAC

| 模型模式 | 步数 (Steps) | NAR 耗时 | VAE 耗时 | 总耗时 | 实时因子 (RTF) | 速度相对实时 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Stock NAR (基座)** | 32 | 5.28 s | 0.30 s | 5.58 s | 0.558 | **1.79× 实时** |
| **Adapted NAR (v4)** | 32 | 9.40 s | 0.34 s | 9.74 s | 0.974 | **1.03× 实时** |
| **Adapted NAR (v4)** | **8 (快速)** | **2.49 s** | **0.33 s** | **2.82 s** | **0.282** | **3.54× 实时** |

### 2. 35.0 秒跨块音频测试 (`reports/realaudio-v4/benchmark_35s`)
- 跨越 MERT 30 秒分块与 512 帧窗口拼接边界；编码耗时：**0.29 s**，生成 875 个语义 token
- 输出音频时长：34.998 秒，完整无截断，峰值 0.978，RMS 0.153

| 模型模式 | 步数 (Steps) | NAR 耗时 | VAE 耗时 | 总耗时 | 实时因子 (RTF) | 速度相对实时 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Stock NAR (基座)** | 32 | 18.46 s | 1.15 s | 19.61 s | 0.560 | **1.78× 实时** |
| **Adapted NAR (v4)** | 32 | 25.70 s | 1.43 s | 27.13 s | 0.775 | **1.29× 实时** |
| **Adapted NAR (v4)** | **8 (快速)** | **6.74 s** | **1.47 s** | **8.21 s** | **0.234** | **4.26× 实时** |

- 试听网页已构建生成：[benchmark_10s/index.html](benchmark_10s/index.html) 与 [benchmark_35s/index.html](benchmark_35s/index.html)。

---

## 四、上游功能支持矩阵与边界说明

| 上游能力 / 脚本 | 本地 MLX 支持状态 | 说明与使用方式 |
| :--- | :---: | :--- |
| `tokenizer_head_joint_v4.pt` (音频编码) | **完全支持 (MLX)** | 通过 `mlx-yue realaudio tokenize`，无 Torch 依赖 |
| `nar_lora_joint_v4.pt` (声学 LoRA) | **完全支持 (MLX)** | 通过 `NARLoRAAdapter` 内存注入，支持 scale 调节 |
| 真实音频端到端重建 | **完全支持 (MLX)** | 通过 `mlx-yue realaudio reconstruct`，支持 32/8 步 |
| 试听与性能对比测试 | **完全支持 (MLX)** | 通过 `mlx-yue realaudio compare` 自动输出报告与网页 |
| 用户自训 Artist AR LoRA 加载 | **支持 (MLX)** | 支持在 BF16 下通过 `ARLoRAAdapter` 融合已训练权重 |
| 上游大规模联合训练 (`joint.py`, `train_v2.py`) | **环境依赖 (CUDA)** | 上游代码硬编码 CUDA、pod 路径及多卡梯度，本地定位为纯推理引擎 |

---

## 五、状态更新与交接

已按方案完成所有实施与验证内容。准备根据用户指示切回 `gpt-6-astra` 进行独立审查与验收。
