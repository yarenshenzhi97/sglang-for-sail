# PPU Candidate Cases

This directory holds PPU test cases that are being prepared for possible CI
workflow integration. Cases here are not discovered or executed by the current
PPU workflow.

## Directory Layout

```text
candidate_case/
└── light_perf/
    ├── deepseek-ai/
    ├── MiniMaxAI/
    ├── moonshotai/
    ├── Qwen/
    └── zai-org/
```

Each model YAML groups its light-performance cases. Keep vendor and model
directories consistent with `test/ppu/recipes/`.

## Case Structure

Candidate YAMLs follow the recipe-style organization used by PPU recipes:

- Top-level `ppu` and `model` identify the framework and model.
- Each item in `light_performance_cases` has its own `ppu` assets and topology,
  `model.base_args` and `model.base_env`, precision under `variants`, and
  benchmark settings under `benchmark`.
- `benchmark.engine` selects the SGLang benchmark entry point. `benchmark.input`
  stores batch size, input lengths, and output length. Optional dataset settings
  and run names are kept with the benchmark.
- `source` points to the originating case description and section line.

Model checkpoints use symbolic `MODEL_PATH_*` aliases. A workflow that consumes
these candidates must map those aliases to checkpoint locations and provide any
required runtime environment before launching a benchmark.

## Case Inventory

The directory currently contains 13 model YAML files and 144 cases. Case IDs
below follow the YAML order; repeated IDs are listed each time they occur in the
source file.

### MiniMax-M3.1-Flash (8 cases)

- `minimax_m3_1_flash_w8a8_int8_tp_prefill`
- `minimax_m3_1_flash_w8a8_int8_tp_decode`
- `minimax_m3_1_flash_w8a8_int8_ep_prefill`
- `minimax_m3_1_flash_w8a8_int8_ep_decode`
- `minimax_m3_1_flash_fp8_channel_tp_prefill`
- `minimax_m3_1_flash_fp8_channel_tp_decode`
- `minimax_m3_1_flash_fp8_channel_ep_prefill`
- `minimax_m3_1_flash_fp8_channel_ep_decode`

### MiniMax-M3 (20 cases)

- `minimax_m3_bf16_tp_prefill`
- `minimax_m3_bf16_tp_decode`
- `minimax_m3_bf16_ep_prefill`
- `minimax_m3_bf16_ep_decode`
- `minimax_m3_w8a8_int8_tp_prefill`
- `minimax_m3_w8a8_int8_tp_decode`
- `minimax_m3_w8a8_int8_ep_prefill`
- `minimax_m3_w8a8_int8_ep_decode`
- `minimax_m3_bf16_tp_prefill`
- `minimax_m3_bf16_tp_decode`
- `minimax_m3_bf16_ep_prefill`
- `minimax_m3_bf16_ep_decode`
- `minimax_m3_fp8_channel_tp_prefill`
- `minimax_m3_fp8_channel_tp_decode`
- `minimax_m3_fp8_channel_ep_prefill`
- `minimax_m3_fp8_channel_ep_decode`
- `minimax_m3_mxfp4_fp8_tp_prefill`
- `minimax_m3_mxfp4_fp8_tp_decode`
- `minimax_m3_mxfp4_fp8_ep_prefill`
- `minimax_m3_mxfp4_fp8_ep_decode`

### Qwen3.8-2.4T-A95B (10 cases)

- `qwen3_8_2_4t_a95b_w8a8_int8_tp32_prefill`
- `qwen3_8_2_4t_a95b_w8a8_int8_tp32_decode`
- `qwen3_8_2_4t_a95b_w4a8_int8_tp32_prefill`
- `qwen3_8_2_4t_a95b_w4a8_int8_tp32_decode`
- `qwen3_8_2_4t_a95b_fp8_tp32_ep32_prefill`
- `qwen3_8_2_4t_a95b_fp8_tp32_ep32_decode`
- `qwen3_8_2_4t_a95b_fp8_tp8_pp4_prefill`
- `qwen3_8_2_4t_a95b_fp8_tp8_pp4_decode`
- `qwen3_8_2_4t_a95b_mxfp4_fp8_tp16_prefill`
- `qwen3_8_2_4t_a95b_mxfp4_fp8_tp16_decode`

### Qwen3.8-27B (6 cases)

- `qwen3_8_27b_bf16_tp1_prefill`
- `qwen3_8_27b_bf16_tp1_decode`
- `qwen3_8_27b_w8a8_int8_tp1_prefill`
- `qwen3_8_27b_w8a8_int8_tp1_decode`
- `qwen3_8_27b_fp8_tp1_prefill`
- `qwen3_8_27b_fp8_tp1_decode`

### Qwen3.8-Flash-Next (14 cases)

- `qwen3_8_next_flash_bf16_tp_prefill`
- `qwen3_8_next_flash_bf16_tp_decode`
- `qwen3_8_next_flash_w8a8_int8_tp_prefill`
- `qwen3_8_next_flash_w8a8_int8_tp_decode`
- `qwen3_8_next_flash_bf16_tp_prefill`
- `qwen3_8_next_flash_bf16_tp_decode`
- `qwen3_8_next_flash_fp8_tp_prefill`
- `qwen3_8_next_flash_fp8_tp_decode`
- `qwen3_8_next_flash_fp8_ep_prefill`
- `qwen3_8_next_flash_fp8_ep_decode`
- `qwen3_8_next_flash_fp8_channel_tp_prefill`
- `qwen3_8_next_flash_fp8_channel_tp_decode`
- `qwen3_8_next_flash_mxfp4_fp8_tp_prefill`
- `qwen3_8_next_flash_mxfp4_fp8_tp_decode`

### Qwen3.8-Max (6 cases)

- `qwen3_8_max_w4a8_int8_tp8_pp4_prefill`
- `qwen3_8_max_w4a8_int8_tp8_pp4_decode`
- `qwen3_8_max_mxfp4_fp8_tp32_ep32_prefill`
- `qwen3_8_max_mxfp4_fp8_tp32_ep32_decode`
- `qwen3_8_max_mxfp4_fp8_tp8_pp2_prefill`
- `qwen3_8_max_mxfp4_fp8_tp8_pp2_decode`

### DeepSeek-V4-Flash (6 cases)

- `dpsk_v4_flash_w8a8_int8_tp_prefill`
- `dpsk_v4_flash_w8a8_int8_tp_decode`
- `dpsk_v4_flash_w4a8_int8_tp_prefill`
- `dpsk_v4_flash_w4a8_int8_tp_decode`
- `dpsk_v4_flash_mxfp4_fp8_tp_prefill`
- `dpsk_v4_flash_mxfp4_fp8_tp_decode`

### DeepSeek-V4-Pro (6 cases)

- `dpsk_v4_pro_w8a8_int8_tp_prefill`
- `dpsk_v4_pro_w8a8_int8_tp_decode`
- `dpsk_v4_pro_w4a8_int8_tp_prefill`
- `dpsk_v4_pro_w4a8_int8_tp_decode`
- `dpsk_v4_pro_mxfp4_fp8_tp_prefill`
- `dpsk_v4_pro_mxfp4_fp8_tp_decode`

### DeepSeek-V4.1-Flash (4 cases)

- `dpsk_v4_1_flash_w8a8_int8_tp_prefill`
- `dpsk_v4_1_flash_w8a8_int8_tp_decode`
- `dpsk_v4_1_flash_mxfp4_fp8_tp_prefill`
- `dpsk_v4_1_flash_mxfp4_fp8_tp_decode`

### Kimi-K3.1-Pro (10 cases)

- `kimi_k3_1_pro_mxfp4_tp_prefill`
- `kimi_k3_1_pro_mxfp4_tp_decode`
- `kimi_k3_1_pro_mxfp4_ep_prefill`
- `kimi_k3_1_pro_mxfp4_ep_decode`
- `kimi_k3_1_pro_mxfp4_pp_prefill`
- `kimi_k3_1_pro_mxfp4_pp_decode`
- `kimi_k3_1_pro_mxfp4_ep_prefill`
- `kimi_k3_1_pro_mxfp4_ep_decode`
- `kimi_k3_1_pro_mxfp4_pp_prefill`
- `kimi_k3_1_pro_mxfp4_pp_decode`

### Kimi-K3 (24 cases)

- `kimi_k3_w4a8_int8_tp_prefill`
- `kimi_k3_w4a8_int8_tp_decode`
- `kimi_k3_w4a8_int8_ep_prefill`
- `kimi_k3_w4a8_int8_ep_decode`
- `kimi_k3_w4a8_int8_pp_prefill`
- `kimi_k3_w4a8_int8_pp_decode`
- `kimi_k3_int4_tp_prefill`
- `kimi_k3_int4_tp_decode`
- `kimi_k3_int4_ep_prefill`
- `kimi_k3_int4_ep_decode`
- `kimi_k3_int4_pp_prefill`
- `kimi_k3_int4_pp_decode`
- `kimi_k3_mxfp4_fp8_tp_prefill`
- `kimi_k3_mxfp4_fp8_tp_decode`
- `kimi_k3_mxfp4_fp8_ep_prefill`
- `kimi_k3_mxfp4_fp8_ep_decode`
- `kimi_k3_mxfp4_fp8_pp_prefill`
- `kimi_k3_mxfp4_fp8_pp_decode`
- `kimi_k3_mxfp4_tp_prefill`
- `kimi_k3_mxfp4_tp_decode`
- `kimi_k3_mxfp4_ep_prefill`
- `kimi_k3_mxfp4_ep_decode`
- `kimi_k3_mxfp4_pp_prefill`
- `kimi_k3_mxfp4_pp_decode`

### GLM-5.3-Flash (18 cases)

- `glm_5_3_flash_bf16_tp_prefill`
- `glm_5_3_flash_bf16_tp_decode`
- `glm_5_3_flash_w8a8_int8_tp_prefill`
- `glm_5_3_flash_w8a8_int8_tp_decode`
- `glm_5_3_flash_w8a8_int8_ep_prefill`
- `glm_5_3_flash_w8a8_int8_ep_decode`
- `glm_5_3_flash_w4a8_int8_tp_prefill`
- `glm_5_3_flash_w4a8_int8_tp_decode`
- `glm_5_3_flash_fp8_tp_prefill`
- `glm_5_3_flash_fp8_tp_decode`
- `glm_5_3_flash_fp8_channel_tp_prefill`
- `glm_5_3_flash_fp8_channel_tp_decode`
- `glm_5_3_flash_fp8_channel_ep_prefill`
- `glm_5_3_flash_fp8_channel_ep_decode`
- `glm_5_3_flash_mxfp4_fp8_tp_prefill`
- `glm_5_3_flash_mxfp4_fp8_tp_decode`
- `glm_5_3_flash_mxfp4_fp8_ep_prefill`
- `glm_5_3_flash_mxfp4_fp8_ep_decode`

### GLM-5.3 (12 cases)

- `glm_5_3_w8a8_int8_tp_prefill`
- `glm_5_3_w8a8_int8_tp_decode`
- `glm_5_3_w4a8_int8_tp_prefill`
- `glm_5_3_w4a8_int8_tp_decode`
- `glm_5_3_fp8_channel_tp_prefill`
- `glm_5_3_fp8_channel_tp_decode`
- `glm_5_3_fp8_channel_ep_prefill`
- `glm_5_3_fp8_channel_ep_decode`
- `glm_5_3_mxfp4_fp8_tp_prefill`
- `glm_5_3_mxfp4_fp8_tp_decode`
- `glm_5_3_mxfp4_fp8_ep_prefill`
- `glm_5_3_mxfp4_fp8_ep_decode`
