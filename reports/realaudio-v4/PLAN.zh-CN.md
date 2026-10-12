# Real-audio tokenizer v4：MLX 移植与集成执行方案

日期：2026-09-15。项目：`/Users/vanch/yue2-mlx`。
当前阶段：完成源码调查，等待 Gemini 实施；功能尚未实现、验收尚未开始。

## 目标与执行模型

将 `Mothersuperior/yue2-mothersuperior-realaudio-tokenizer-v4` 的真实录音语义编码器与 NAR 适配器转换为可复现的 MLX 权重，接入项目 CLI/API 和项目、本地 yue2-music 技能，实际运行编码与重建并提供试听、时间、内存、RTF 和数值证据。

用户指定：本任务后续实施使用 `google-antigravity/gemini-3.8-flash`，实施结束后使用 `gpt-6-astra` 独立验收。通过应用 `send_message_to_thread` 对本任务发送指定 model 的后续消息完成路由；必须以实际工具响应及后续执行证据为准，不声称仅写入模型名称就已经切换。目标保持 active，直至 Astra 验收和必要修复完成。

## 已确认的源码事实

- 上游固定 revision：`f2278a2e005dc4ecc421c53a0929f62b3aeb2280`。
- 源码快照与 SHA256 清单：`models/realaudio-v4/upstream/source-manifest.json`（本地忽略目录）。最终报告另保存可提交的来源清单，不提交大权重。
- `tokenizer_head_joint_v4.pt`：171,305,291 字节；SHA256 `d23c4f757a05f031134b8471ec84245ec2338966516e1a9e26a17ff300a5f87e`。
- `nar_lora_joint_v4.pt`：140,626,123 字节；SHA256 `df175dbf9405a8e15b2c3f8dbdcc97303575f763787f227b03029020e28102fe`。
- 编码器：MERT-v2-FullSong `hidden_states[20]`，输入维度 1024，投影到 512；8 层、8 头、pre-norm Transformer、GELU、FFN 2048、可学习 512 帧位置编码、32768 分类输出。
- 音频处理：24 kHz 单声道，MERT 按 30 秒分组处理；原脚本舍弃不足 1 秒的尾块，再按整曲时长线性重采样到 25 Hz（`align_corners=False`）。保留这一行为的兼容模式，明确处理不足 1 秒输入，不静默丢弃整曲或换成 SheetSage 层加权特征。
- `ar_prep.py` 的实例归一化在整曲上进行，float32、总体标准差 `ddof=0`、epsilon `1e-5`。不同训练脚本有 float16 中间舍入，必须按使用场景记录精度，不能混用。
- 预测窗口 512、步长 256；短窗口补零；中间窗口各裁掉 128 帧，最后额外贴尾窗口按源码顺序写回。测试非整窗长度的覆盖、重叠与边界。
- NAR LoRA：28 层，每层 q/k/v/o 与 gate/up/down 共 7 个线性层，196 对 A/B；`W += scale * (B @ A)`，没有隐含 alpha/rank 缩放；完整替换 `vae2llm`、`llm2vae` 权重。AR LoRA 使用同类列表顺序，需按真实 checkpoint 的 rank/shape 校验。
- 当前 MLX MERT 只返回末层或层加权值，需要兼容地增加指定中间层提取；当前 VAE 只有 decoder，不能把它当成可训练预处理所需的 encoder。
- 原仓库还包含联合训练、AR artist LoRA、Demucs/MMS 对齐、regularizer pack 等脚本，必须在功能矩阵逐项说明本地支持情况；这份模型不是已经训练好的用户专属 artist AR LoRA。
- 来源：[模型卡](https://huggingface.co/Mothersuperior/yue2-mothersuperior-realaudio-tokenizer-v4)。模型衍生权重许可 CC BY-NC 4.0，保留来源与许可。

## 实施顺序与交付

### 1. 基线、来源与转换

1. 复核 git 状态、项目说明、运行环境和已有测试；保留用户已有修改。
2. 使用固定 revision 下载上述两份权重，核对字节数/SHA256；复用现有 MERT/YuE2/VAE，不复制整套基础模型。
3. 在隔离的 `.oracle` 或开发环境内用 `torch.load(..., weights_only=True, map_location="cpu")` 读取转换输入；拒绝不明对象、缺键、多键、错误 rank/shape、NaN/Inf。生产 `.runtime` 不依赖 Torch，不直接运行有 CUDA/绝对路径副作用的上游训练脚本。
4. 新增独立转换模块与 CLI，输出 head、NAR LoRA 的 safetensors/config/manifest；准确记录源 revision、hash、dtype、映射、依赖版本及许可证。先 FP32 head 基线，再测 BF16；不擅自量化分类 head。

### 2. MLX 真实音频编码

1. 为 MERT 增加精确的 hidden-state 层选择接口，确认编号与上游一致，不改变现有转谱默认行为。
2. 实现原生 MLX tokenizer head，包括 PyTorch packed QKV 拆分、线性权重布局、LayerNorm epsilon、精确 GELU、双向注意力、位置编码和 eval/dropout 语义。
3. 实现输入读取、重采样、30 秒 MERT 分组、25 Hz 插值、整曲归一化、512 帧窗口拼接；明确空文件、极短、静音、立体声与多声道行为。
4. 保存 semantic IDs（0..32767）、时长、输入 hash、有效配置、各阶段耗时、模型标识；与项目协议中的 token offset 分开处理。

### 3. NAR/AR 适配器与真实录音重建

1. 以可选适配器接入 pipeline，默认行为保持兼容；验证全量 I/O 层覆盖。基础磁盘权重不可原地改写，禁止多次加载累加同一增量。
2. 支持 NAR scale 与有文件时的 AR LoRA scale。明确 NAR 的 I/O 替换在 scale=0 时如何处理，并提供真正的 stock 模式，不能把 scale=0 自动当成未加载。
3. AR 8bit 加载时采用正确的浮点基座合并后再量化，或验证过的旁路 LoRA；不能对打包整数权重直接加 delta。
4. 提供真实录音 → semantic tokens → NAR+VAE 的重建 CLI/API；允许保存 token 后离线重放、同 seed/noise 的 stock/adapted 与 8/32 步比较。
5. 集成新歌、已有五种模式和 melody cover 的 adapter 参数；保存的 plan/result/replay 配置包含适配器 hash/scale，避免重放丢失设置。不要把录音 token 重建命名为音色克隆或保证保留全部原曲细节。

### 4. 训练功能范围核对

按仓库真实能力列出数据预处理、tokenize 数据集、NAR/head 联合适配、AR LoRA、lyric-cursor、regularizer 与 checkpoint ladder 的支持矩阵。

本次核心可验收结果必须是两份已发布权重的 MLX 转换、编码与适配重建及项目/技能集成。对于上游训练功能，优先补充可用的数据准备/适配器导入接口与准确文档；如果扩展本机训练，则必须实现实际梯度、冻结参数、checkpoint 恢复与小样本训练验证，不能以占位 CLI 或调用 CUDA 脚本宣称 MLX 训练完成。未移植的训练步骤明确列为未支持，不能宣称整个训练仓库已经全功能移植。没有用户训练语料时，不捏造 artist 训练质量结果。

### 5. 技能、文档与本地同步

1. 先更新 `skills/yue2-music/reports/skill-ir.json` 的职责、触发/非触发、资源、证据与限制，保留现有 owner/maturity 信息。
2. 扩展 SKILL.md 和能力表，新增 real-audio 使用说明/包装脚本；让用户可以请求真实音频编码、重建、NAR 适配、导入 artist AR LoRA、质量速度对比，并根据需求选取参数。
3. 明确“音频转乐谱 cover”和“音频转 semantic tokens 重建”的区别；保留完整时长、标准 32 步/快听 8 步默认值。参数依据实测，不能称存在普适最佳设置。
4. 更新项目 README、模型设置、示例与功能矩阵。验证脚本参数完整传递、离线模式、路径含空格、错误输入、取消和资源监控。
5. 本地 canonical 是 `/Users/vanch/.skills-manager/skills/yue2-music`，Codex 为指向它的软链接；先比对并备份需修改文件，再同步项目与本地技能，核对文件 hash。其他环境的现有链接维持一致。

### 6. 数值与真实运行验收证据

- 单测：权重转换全覆盖、QKV、归一化、窗口边界（含 1/511/512/513/768/769 帧）、LoRA 合并/scale/卸载/重复加载、config/replay、异常输入。
- Oracle：从已审阅源码抽取无顶层副作用的 PyTorch 参考；相同实际 checkpoint/固定输入逐项比较 MERT L20、head 层输出/logits/top-1、LoRA delta 与 NAR velocity/短程 solve。FP32 和 BF16 分开报告；在看到候选结果前记录数值阈值，不能事后调宽以通过。
- 区分“MLX 相对 Torch token 一致率”与“模型预测原始 YuE 生成 tokens 的正确率”，不套用作者约 95% 主观试听宣传数字。现有基础模型数值偏差须通过 stock 对照定位，不能掩盖或无关扩大修复。
- 真实运行至少覆盖短片、跨 30 秒/512 帧边界及一个完整音乐文件；复用许可明确的本地测试样本，全部记录输入 hash。整曲不得使用 800-token 截断冒充；重建长度按编码帧数处理。
- 试听报告包含原录音、stock NAR、adapted 32 步、adapted 8 步；记录编码/加载/NAR/VAE/总耗时、音频长度、RTF 分母与计时边界、MLX peak 和进程峰值内存（区分单位）。做有限性/非静音/削波/时长核对；未实际听评就不写主观质量通过。
- 复用现有资源 watchdog、取消检查，单 GPU 任务串行；检查接电条件。控制输出规模，只保留必要音频和小型证据，不恢复此前已清理的大量冗余文件。
- 运行相关新测试、既有回归、lint 和 skill contract/routing 校验；记录真实命令、返回码及失败。

## Gemini → Astra 交接要求

实施过程中维护 `reports/realaudio-v4/STATUS.md`，记录阶段、改动、检查、证据和下一步，不写空泛完成百分比。最终提供 `REPORT.zh-CN.md`、机器可读指标、可试听音频和功能矩阵。

Gemini 完成实现后必须调用 `send_message_to_thread`，对当前任务 `01a09a26-db6c-77d3-9e6c-062ea6ba73eb` 指定 `model="gpt-6-astra"`，请求独立检查 diff、转换 provenance、关键数值、实际 CLI、技能同步、回归结果并修复必需缺陷。不要另开任务或调用不支持 Gemini 的 collaboration agent 代替用户要求的切换。

Astra 只有在核心交付和必要验证成立后才将对话目标设为 complete。路由失败要准确报告，并保留实施恢复点；不能悄悄替换用户指定执行模型。没有额外用户指令时不向 GitHub/Hugging Face 发布新模型，也不触碰社交账号。
