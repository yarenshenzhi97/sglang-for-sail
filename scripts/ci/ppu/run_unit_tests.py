#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
TESTS = """
test/registered/unit/cp/test_cp_interleave_order.py
test/registered/unit/disaggregation/test_dcp_pack.py
test/registered/unit/disaggregation/test_disaggregation_wire.py
test/registered/unit/disaggregation/test_kv_transfer_replica_metric.py
test/registered/unit/disaggregation/test_mooncake_efa_allocator.py
test/registered/unit/disaggregation/test_mooncake_transfer_batching.py
test/registered/unit/disaggregation/test_pp_hybrid_kv_transfer.py
test/registered/unit/distributed/test_custom_all_reduce_v2_capability.py
test/registered/unit/entrypoints/openai/test_protocol.py
test/registered/unit/entrypoints/test_grpc_bridge.py
test/registered/unit/entrypoints/test_server_info.py
test/registered/unit/function_call/test_kimik3_structural_tag.py
test/registered/unit/function_call/test_normalize_json_schema_types.py
test/registered/unit/kernels/test_dsa_fused_store_ffi.py
test/registered/unit/kernels/test_fp8_quantization.py
test/registered/unit/layer_boundary/test_kimi_k3_stage_kernels.py
test/registered/unit/layers/attention/test_ascend_kda_extend_contract.py
test/registered/unit/layers/attention/test_dsa_backend_dp_padding.py
test/registered/unit/layers/attention/test_dsa_indexer_idle.py
test/registered/unit/layers/attention/test_dsa_tilelang_v1_gemm_policy.py
test/registered/unit/layers/attention/test_dsa_topk_ppu_dtype.py
test/registered/unit/layers/attention/test_ppu_flash_mla_wrapper.py
test/registered/unit/layers/attention/test_qsa_decode_head_padding.py
test/registered/unit/layers/moe/test_ppu_kimi_k3_moe.py
test/registered/unit/layers/moe/test_ppu_kimi_k3_w4a8.py
test/registered/unit/layers/quantization/test_ppu_bf16_dense_deepgemm.py
test/registered/unit/layers/quantization/test_ppu_mxfp4_w4a16.py
test/registered/unit/layers/quantization/test_w8a8_int8.py
test/registered/unit/layers/test_deepep_buffer_layouts.py
test/registered/unit/layers/test_deepep_graph_adapter.py
test/registered/unit/layers/test_dsv41_candidate_blocks.py
test/registered/unit/layers/test_engram_load_rows.py
test/registered/unit/layers/test_radix_attention_padded_extend.py
test/registered/unit/managers/test_dcp_logical_capacity.py
test/registered/unit/managers/test_pp_cp_rank_offsets.py
test/registered/unit/mem_cache/test_allocation_sizing.py
test/registered/unit/mem_cache/test_dsa_pool_host_unit.py
test/registered/unit/mem_cache/test_fused_draft_placement.py
test/registered/unit/mem_cache/test_hicache_dcp_host_pool.py
test/registered/unit/model_executor/runner/test_dsv41_draft_metadata_glue.py
test/registered/unit/model_executor/test_pool_configurator.py
test/registered/unit/model_executor/test_ppu_draft_metadata_padding.py
test/registered/unit/model_executor/test_prefill_cuda_graph_runner.py
test/registered/unit/model_executor/test_prefill_cuda_graph_runner_helpers.py
test/registered/unit/models/test_deepseek_v41_tail_cp_ids.py
test/registered/unit/models/test_deepseek_v4_wo_a_streaming.py
test/registered/unit/models/test_minimax_m3.py
test/registered/unit/multimodal/test_base_processor_image_decode.py
test/registered/unit/observability/test_scheduler_stage_metrics.py
test/registered/unit/server_args/test_cp_tp_group_sharing_hook.py
test/registered/unit/spec/test_dflash_draft_graph_eligibility.py
test/registered/unit/spec/test_dflash_logits.py
test/registered/unit/spec/test_dflash_mla.py
test/registered/unit/spec/test_dflash_pd_dcp.py
test/registered/unit/spec/test_oot_dflash_hooks.py
test/registered/unit/test_model_overrides.py
test/registered/kernels/ops/attention/dsv4/test_index_k_norm_rope_pack.py
test/registered/kernels/ops/attention/dsv4/test_int8_indexer.py
test/registered/kernels/ops/attention/dsv4/test_v41_kv_layout_arrangement.py
test/registered/kernels/ops/attention/test_attn_residual_sm8x.py
test/registered/kernels/ops/attention/test_concat_mla.py
test/registered/kernels/ops/attention/test_dcp_lse_combine.py
test/registered/kernels/ops/attention/test_dcp_push_combine.py
test/registered/kernels/ops/attention/test_dsa_dcp_topk_merge.py
test/registered/kernels/ops/attention/test_dsa_indexer.py
test/registered/kernels/ops/attention/test_dsa_kpool_multi_pool.py
test/registered/kernels/ops/attention/test_dsa_metadata.py
test/registered/kernels/ops/attention/test_dsa_multi_ctas_counter.py
test/registered/kernels/ops/attention/test_kda_replayssm_ring_fused.py
test/registered/kernels/ops/attention/test_minimax_m3_msa_ppu_page_table_tail.py
test/registered/kernels/ops/attention/test_mla_window_indices.py
test/registered/kernels/ops/attention/test_topk_v1.py
test/registered/kernels/ops/embeddings/test_engram_gather_sorted.py
test/registered/kernels/ops/gemm/test_ppu_bf16_dense_gemm.py
test/registered/kernels/ops/memory/test_interleave_rows.py
test/registered/kernels/ops/moe/test_mega_moe_pre_dispatch.py
test/registered/kernels/ops/quantization/test_fp8_kernel.py
test/registered/kernels/ops/speculative/test_dflash_grouped_conv.py
test/registered/dcp/test_dcp_layout_unit.py
test/registered/function_call/test_kimik3_detector.py
test/registered/spec/dspark/test_dspark_draft_path_default.py
test/registered/spec/dspark/test_dspark_kernel_parity.py
test/registered/spec/dspark/test_ragged_verify.py
test/ppu/unit/test_deep_gemm_fp8_mqa_avg_logits_d32.py
test/ppu/unit/test_deepseek_v4_dspark_quant_metadata.py
test/ppu/unit/test_dsa_fused_indexer_wk_loader.py
test/ppu/unit/test_dsv41_mega_mhc_ppu.py
test/ppu/unit/test_dsv41_mhc_prenorm_ppu.py
test/ppu/unit/test_dsv41_sparse_indexer_support.py
test/ppu/unit/test_dsv4_topk_deep_select.py
test/ppu/unit/test_glm53_migration_contracts.py
test/ppu/unit/test_indexer_api_compat.py
test/ppu/unit/test_pla_adaptation.py
test/ppu/unit/test_qsa_ppu_backends.py
test/ppu/unit/test_qsa_ppu_compressed_logits.py
test/ppu/unit/test_sail_moe_tp_fused_default.py
test/ppu/unit/test_topk_prefill_bf16_alignment.py
test/ppu/module/test_deepgemm_moe_w4a8.py
test/ppu/module/test_optimal_fusedmoe.py
test/ppu/module/test_ppu_deepep_hooks.py
test/manual/ppu/test_dflash_flashmla_swa.py
python/sglang/kernels/aot/tests/test_flashmla_topk_length_contract.py
python/sglang/kernels/aot/tests/test_int8_gemm.py
""".split()


def _has_custom_entry_point(path: str) -> bool:
    return "__main__" in (REPO / path).read_text(encoding="utf-8")


def main() -> int:
    available = [path for path in TESTS if (REPO / path).is_file()]
    missing = [path for path in TESTS if not (REPO / path).is_file()]
    if missing:
        print(
            "Unit-test manifest contains paths absent from this checkout; "
            "skipping them:\n" + "\n".join(f"  {path}" for path in missing),
            file=sys.stderr,
        )
    if not available:
        print("No unit-test files from the manifest exist in this checkout", file=sys.stderr)
        return 2

    env = os.environ.copy()
    source_path = str(REPO / "python")
    env["PYTHONPATH"] = source_path + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    env["PYTHONUNBUFFERED"] = "1"

    failures = []
    for index, path in enumerate(available, start=1):
        if _has_custom_entry_point(path):
            command = [sys.executable, path]
        else:
            command = [sys.executable, "-m", "pytest", "-q", path]
        print(f"\n[{index}/{len(available)}] RUN {path}", flush=True)
        status = subprocess.run(
            command, cwd=REPO, env=env, check=False
        ).returncode
        if status:
            failures.append((path, status))
            print(f"[{index}/{len(available)}] FAIL {path} (exit {status})", flush=True)
        else:
            print(f"[{index}/{len(available)}] PASS {path}", flush=True)

    print(
        f"\nUnit-test files: {len(available) - len(failures)} passed, "
        f"{len(failures)} failed, {len(missing)} missing"
    )
    if failures:
        print("Failed tests:", file=sys.stderr)
        for path, status in failures:
            print(f"  {path} (exit {status})", file=sys.stderr)
        return failures[0][1]
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
