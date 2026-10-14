# Third-party code notices

The `lyra.music_tools.compile_score`, `instrumentalize` and `instrumental`
helpers are adapted from the official YuE repository's instrumental skill at
commit `1647252d68b70cbead046ffd3f7caf036dde47ee`:
https://github.com/multimodal-art-projection/YuE/tree/1647252d68b70cbead046ffd3f7caf036dde47ee/skills/yue2-music/instrumental/scripts

- Copyright (c) 2026 yue2-music contributors, MIT.
  Full text: `licenses/yue2-instrumental-MIT.txt`.
- Adaptations use package-relative imports and the native MLX generation and
  transcription APIs; the symbolic event compiler and voice-transfer policy
  retain the original exact-fraction invariants.

The Oobleck VAE and SnakeBeta implementation in `modeling_vae.py` is derived
from stable-audio-tools commit `a6ae0cdf8b2eb1567a4b42ceadddec3712d99d45`.
The module hierarchy, weight normalization and activation equations preserve
the checkpoint's original inference implementation.

- Oobleck / stable-audio-tools: Copyright (c) 2023 Stability AI, MIT.
  Full text: `licenses/stable-audio-tools-MIT.txt`.
- SnakeBeta / BigVGAN: Copyright (c) 2022 NVIDIA CORPORATION, MIT.
  Full text: `licenses/SnakeBeta-NVIDIA-MIT.txt`.

These notices cover the identified source code and retain its original licenses.
The YuE2 model checkpoint weights are separately licensed under CC BY-NC 4.0;
see MODEL_LICENSE for the scope and full terms. This does not relicense third-party code.
