from __future__ import annotations

import pytest

from vllm_omni.metrics import OrchestratorAggregator
from vllm_omni.metrics.stats import RequestE2EStats, StageRequestStats, StageStats

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _get_request_entry(table: list[dict], request_id: str) -> dict:
    for entry in table:
        if entry.get("request_id") == request_id:
            return entry
    raise AssertionError(f"request_id={request_id} not found")


def test_orchestrator_aggregator_builds_summary(monkeypatch: pytest.MonkeyPatch) -> None:
    agg = OrchestratorAggregator(num_stages=2, log_stats=True, wall_start_ts=0.0, final_stage_id_for_e2e=1)
    agg.stage_first_ts[0] = 0.0
    agg.stage_last_ts[0] = 0.03
    agg.stage_first_ts[1] = 0.05
    agg.stage_last_ts[1] = 0.07

    agg.on_forward(0, 1, "r1", size_bytes=1024, tx_ms=5.0, used_shm=False)

    agg.on_stage_metrics(
        0,
        "r1",
        StageRequestStats(
            batch_id=1,
            batch_size=1,
            num_tokens_in=3,
            num_tokens_out=3,
            stage_gen_time_ms=30.0,
            rx_transfer_bytes=0,
            rx_decode_time_ms=0.0,
            rx_in_flight_time_ms=0.0,
            stage_stats=StageStats(),
            handoff_to_stage_id=1,
            stage_handoff_time_ms=20.0,
            ar2diffusion_time_ms=12.5,
            output_processor_time_ms=7.0,
        ),
    )
    agg.on_stage_metrics(
        1,
        "r1",
        StageRequestStats(
            batch_id=1,
            batch_size=1,
            num_tokens_in=0,
            num_tokens_out=4,
            stage_gen_time_ms=20.0,
            rx_transfer_bytes=1024,
            rx_decode_time_ms=5.0,
            rx_in_flight_time_ms=2.0,
            stage_stats=StageStats(),
        ),
    )
    agg.record_final_output_time("r1", 3.0)
    monkeypatch.setattr("vllm_omni.metrics.stats.time.time", lambda: 0.07)
    agg.on_finalize_request(1, "r1", req_start_ts=0.0)

    summary = agg.build_and_log_summary()
    overall = summary["overall_summary"]
    assert overall["num_of_requests"] == 1
    assert overall["request_wall_time_ms"] == 70.0
    assert overall["input_preprocess_time_ms"] == 0.0
    assert overall["stage_gen_total_time_ms"] == 50.0
    assert overall["output_processor_total_time_ms"] == 7.0
    assert overall["stage_handoff_total_time_ms"] == 20.0
    assert overall["final_output_total_time_ms"] == 3.0
    assert overall["stage_0_to_1_handoff_time_ms"] == 20.0
    assert overall["ar2diffusion_total_time_ms"] == 12.5
    assert overall["stage_0_to_1_ar2diffusion_time_ms"] == 12.5

    stage_entry = _get_request_entry(summary["stage_table"], "r1")
    stage_ids = [row["stage_id"] for row in stage_entry["stages"]]
    assert stage_ids == [0, 1]
    assert "stage_handoff_time_ms" not in stage_entry["stages"][0]
    assert "ar2diffusion_time_ms" not in stage_entry["stages"][0]

    transfer_entry = _get_request_entry(summary["trans_table"], "r1")
    assert transfer_entry["transfers"][0]["edge"] == "0->1"
    assert transfer_entry["transfers"][0]["size_kbytes"] == 1.0

    e2e_entry = _get_request_entry(summary["e2e_table"], "r1")
    assert e2e_entry["total_tokens"] == 10
    assert e2e_entry["final_output_time_ms"] == 3.0


def test_build_and_log_summary_e2e_only() -> None:
    agg = OrchestratorAggregator(num_stages=1, log_stats=True, wall_start_ts=0.0, final_stage_id_for_e2e=0)
    agg.e2e_events.append(
        RequestE2EStats(
            request_id="r",
            request_wall_time_ms=10.0,
            input_preprocess_time_ms=0.0,
            engine_pipeline_time_ms=10.0,
            total_tokens=5,
            transfers_total_time_ms=0.0,
            transfers_total_bytes=0,
        )
    )

    summary = agg.build_and_log_summary()
    e2e_entry = _get_request_entry(summary["e2e_table"], "r")
    assert e2e_entry["request_wall_time_ms"] == 10.0
    assert e2e_entry["engine_pipeline_time_ms"] == 10.0
    assert e2e_entry["total_tokens"] == 5
    stage_entry = _get_request_entry(summary["stage_table"], "r")
    assert stage_entry["stages"] == []


def test_build_and_log_summary_multiple_requests() -> None:
    agg = OrchestratorAggregator(
        num_stages=2, log_stats=True, wall_start_ts=0.0, final_stage_id_for_e2e={"r1": 1, "r2": 0}
    )

    # Request r1 goes through both stages
    agg.on_stage_metrics(
        0,
        "r1",
        StageRequestStats(
            batch_id=1,
            batch_size=1,
            num_tokens_in=2,
            num_tokens_out=4,
            stage_gen_time_ms=10.0,
            rx_transfer_bytes=0,
            rx_decode_time_ms=0.0,
            rx_in_flight_time_ms=0.0,
            stage_stats=StageStats(),
        ),
    )
    agg.on_stage_metrics(
        1,
        "r1",
        StageRequestStats(
            batch_id=1,
            batch_size=1,
            num_tokens_in=4,
            num_tokens_out=5,
            stage_gen_time_ms=8.0,
            rx_transfer_bytes=0,
            rx_decode_time_ms=0.0,
            rx_in_flight_time_ms=0.0,
            stage_stats=StageStats(),
        ),
    )
    agg.on_finalize_request(1, "r1", req_start_ts=0.0)

    # Request r2 only goes through stage 0
    agg.on_stage_metrics(
        0,
        "r2",
        StageRequestStats(
            batch_id=2,
            batch_size=1,
            num_tokens_in=1,
            num_tokens_out=2,
            stage_gen_time_ms=12.0,
            rx_transfer_bytes=0,
            rx_decode_time_ms=0.0,
            rx_in_flight_time_ms=0.0,
            stage_stats=StageStats(),
        ),
    )
    agg.on_finalize_request(0, "r2", req_start_ts=0.0)

    summary = agg.build_and_log_summary()
    assert len(summary["stage_table"]) == 2
    assert {entry["request_id"] for entry in summary["e2e_table"]} == {"r1", "r2"}
    assert summary["overall_summary"]["engine_pipeline_time_ms"] > 0.0
    # Check that r1 has two stages and r2 has one
    r1_stage_entry = next(e for e in summary["stage_table"] if e["request_id"] == "r1")
    r2_stage_entry = next(e for e in summary["stage_table"] if e["request_id"] == "r2")
    assert len(r1_stage_entry["stages"]) == 2
    assert len(r2_stage_entry["stages"]) == 1
