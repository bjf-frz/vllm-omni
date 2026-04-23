from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from vllm.logger import init_logger

from vllm_omni.metrics.utils import _build_field_defs, _build_row, _format_table

logger = init_logger(__name__)


@dataclass
class StageStats:
    total_token: int = 0
    total_gen_time_ms: float = 0.0

    @property
    def avg_tokens_per_s(self) -> float:
        return (self.total_token * 1000.0 / self.total_gen_time_ms) if self.total_gen_time_ms > 0 else 0.0


@dataclass
class StageRequestStats:
    batch_id: int
    batch_size: int
    num_tokens_in: int
    num_tokens_out: int
    stage_gen_time_ms: float
    rx_transfer_bytes: int
    rx_decode_time_ms: float
    rx_in_flight_time_ms: float
    stage_stats: StageStats
    stage_id: int | None = None
    stage_name: str | None = None
    stage_type: str | None = None
    final_output_type: str | None = None
    request_id: str | None = None
    postprocess_time_ms: float = 0.0
    diffusion_metrics: dict[str, int] = None
    audio_generated_frames: int = 0
    stage_end_ts: float | None = None
    handoff_to_stage_id: int | None = None
    stage_handoff_time_ms: float = 0.0
    ar2diffusion_time_ms: float = 0.0

    @property
    def rx_mbps(self) -> float:
        return (
            (float(self.rx_transfer_bytes) * 8.0) / (max(float(self.rx_decode_time_ms), 1e-6) * 1000.0)
            if self.rx_transfer_bytes > 0
            else 0.0
        )

    @property
    def tokens_per_s(self) -> float:
        return (self.num_tokens_out * 1000.0 / self.stage_gen_time_ms) if (self.stage_gen_time_ms > 0) else 0.0


@dataclass
class TransferEdgeStats:
    from_stage: int
    to_stage: int
    request_id: str
    size_bytes: int
    tx_time_ms: float
    used_shm: bool = False
    rx_decode_time_ms: float = 0.0
    in_flight_time_ms: float = 0.0

    @property
    def total_time_ms(self) -> float:
        return float(self.tx_time_ms) + float(self.rx_decode_time_ms) + float(self.in_flight_time_ms)


@dataclass
class RequestE2EStats:
    request_id: str
    request_wall_time_ms: float
    input_preprocess_time_ms: float
    engine_pipeline_time_ms: float
    e2e_total_tokens: int
    transfers_total_time_ms: float
    transfers_total_bytes: int

    @property
    def e2e_tpt(self) -> float:
        return (self.engine_pipeline_time_ms / self.e2e_total_tokens) if self.e2e_total_tokens > 0 else 0.0


# === Field Configuration ===
# Fields requiring unit conversion:  original_field_name -> (display_name, transform_fn)
FIELD_TRANSFORMS: dict[str, tuple[str, Callable[[Any], Any]]] = {
    "rx_transfer_bytes": ("rx_transfer_kbytes", lambda v: v / 1024.0),
    "size_bytes": ("size_kbytes", lambda v: v / 1024.0),
    "transfers_total_bytes": ("transfers_total_kbytes", lambda v: v / 1024.0),
}

# Fields to exclude from table display for each event type
STAGE_EXCLUDE = {
    "stage_stats",
    "stage_id",
    "request_id",
    "rx_transfer_bytes",
    "rx_decode_time_ms",
    "rx_in_flight_time_ms",
    "final_output_type",
    "stage_end_ts",
    "handoff_to_stage_id",
    "stage_handoff_time_ms",
    "ar2diffusion_time_ms",
}
TRANSFER_EXCLUDE = {"from_stage", "to_stage", "request_id", "used_shm"}
E2E_EXCLUDE = {"request_id"}

# Decide the order of overall summary fields, or None for auto
OVERALL_FIELDS: list[str] | None = [
    "e2e_requests",
    "request_wall_time_ms",
    "input_preprocess_time_ms",
    "engine_pipeline_time_ms",
    "stage_gen_total_time_ms",
    "stage_handoff_total_time_ms",
    "ar2diffusion_total_time_ms",
    "final_output_overhead_time_ms",
    "e2e_total_tokens",
    "avg_request_wall_time_ms",
    "avg_input_preprocess_time_ms",
    "avg_engine_pipeline_time_ms",
    "avg_stage_gen_total_time_ms",
    "avg_stage_handoff_total_time_ms",
    "avg_ar2diffusion_time_ms",
    "avg_final_output_overhead_time_ms",
    "e2e_avg_tokens_per_s",
]
STAGE_FIELDS = _build_field_defs(StageRequestStats, STAGE_EXCLUDE, FIELD_TRANSFORMS)
TRANSFER_FIELDS = _build_field_defs(TransferEdgeStats, TRANSFER_EXCLUDE, FIELD_TRANSFORMS)
E2E_FIELDS = _build_field_defs(RequestE2EStats, E2E_EXCLUDE, FIELD_TRANSFORMS)


class OrchestratorAggregator:
    def __init__(
        self,
        num_stages: int,
        log_stats: bool,
        wall_start_ts: float,
        final_stage_id_for_e2e: dict[str, int] | int,
        stage_metadata: list[dict[str, Any]] | None = None,
    ) -> None:
        self.num_stages = int(num_stages)
        self.log_stats = bool(log_stats)
        self.final_stage_id_for_e2e = final_stage_id_for_e2e
        self.stage_metadata = list(stage_metadata or [])
        self.init_run_state(wall_start_ts)
        self.stage_events: dict[str, list[StageRequestStats]] = {}
        self.transfer_events: dict[
            tuple[int, int, str], TransferEdgeStats
        ] = {}  # Key: (from_stage, to_stage, request_id)
        self.e2e_events: list[RequestE2EStats] = []

    def init_run_state(self, wall_start_ts: float) -> None:
        # Per-run aggregates and timing state
        self.stage_total_tokens = [0 for _ in range(self.num_stages)]
        self.engine_pipeline_total_ms = 0.0
        self.input_preprocess_total_ms = 0.0
        self.e2e_total_tokens = 0
        self.e2e_count = 0
        self.e2e_done = set()
        self.wall_start_ts = float(wall_start_ts)
        self.last_finish_ts = float(wall_start_ts)
        self.stage_first_ts = [None for _ in range(self.num_stages)]
        self.stage_last_ts = [None for _ in range(self.num_stages)]
        self.accumulated_gen_time_ms: defaultdict[str, defaultdict[int, float]] = defaultdict(
            lambda: defaultdict(float)
        )  # {request_id: {stage_id:accumulated_gen_time_ms}}
        self.diffusion_metrics: defaultdict[str, defaultdict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )  # {request_id: {diffusion_metrics_key: accumulated_metrics_data}}

    def _get_stage_metadata(self, stage_id: int | None) -> dict[str, Any]:
        if stage_id is None or stage_id < 0 or stage_id >= len(self.stage_metadata):
            return {}
        meta = self.stage_metadata[stage_id]
        return dict(meta) if isinstance(meta, dict) else {}

    @staticmethod
    def _get_stage_type(stage_meta: dict[str, Any]) -> str:
        return str(stage_meta.get("stage_type") or "unknown")

    @classmethod
    def _get_stage_name(cls, stage_id: int, stage_meta: dict[str, Any]) -> str:
        stage_type = cls._get_stage_type(stage_meta)
        model_stage = stage_meta.get("model_stage")
        if model_stage:
            return str(model_stage)
        if stage_type == "diffusion":
            return "diffusion"
        return f"stage_{stage_id}"

    def _stage_label(self, stage_id: int | None) -> str:
        if stage_id is None:
            return "unknown"
        stage_meta = self._get_stage_metadata(stage_id)
        return f"{stage_id}:{self._get_stage_name(stage_id, stage_meta)}"

    def _format_seconds(self, ms: float) -> str:
        return f"{ms / 1000.0:.2f}s"

    def _format_ms(self, ms: float) -> str:
        return f"{ms:,.3f} ms"

    def _log_omni_timing(self, evt: RequestE2EStats) -> None:
        rid = evt.request_id
        stages = []
        handoffs = []
        ar2diffusion_ms = 0.0
        for stage_evt in sorted(
            self.stage_events.get(rid, []),
            key=lambda e: e.stage_id if e.stage_id is not None else -1,
        ):
            stages.append(
                f"{self._stage_label(stage_evt.stage_id)}={self._format_seconds(stage_evt.stage_gen_time_ms)}"
            )
            ar2diffusion_ms += float(stage_evt.ar2diffusion_time_ms or 0.0)
            if stage_evt.handoff_to_stage_id is not None and stage_evt.stage_handoff_time_ms > 0.0:
                handoff = (
                    f"{stage_evt.stage_id}->{stage_evt.handoff_to_stage_id}="
                    f"{stage_evt.stage_handoff_time_ms:.3f}ms"
                )
                if stage_evt.ar2diffusion_time_ms > 0.0:
                    handoff += f"(ar2diffusion={stage_evt.ar2diffusion_time_ms:.3f}ms)"
                handoffs.append(handoff)
        transfers = []
        for transfer_evt in sorted(
            [e for e in self.transfer_events.values() if e.request_id == rid],
            key=lambda e: (e.from_stage, e.to_stage),
        ):
            transfers.append(f"{transfer_evt.from_stage}->{transfer_evt.to_stage}={transfer_evt.total_time_ms:.3f}ms")

        ar2diffusion = f" ar2diffusion={self._format_seconds(ar2diffusion_ms)}" if ar2diffusion_ms > 0.0 else ""
        logger.info(
            "[OmniTiming] req=%s total=%s preprocess=%s engine=%s%s stages=[%s] handoffs=[%s] transfers=[%s]",
            rid,
            self._format_seconds(evt.request_wall_time_ms),
            self._format_seconds(evt.input_preprocess_time_ms),
            self._format_seconds(evt.engine_pipeline_time_ms),
            ar2diffusion,
            ",".join(stages),
            ",".join(handoffs),
            ",".join(transfers),
        )

    def _build_timing_breakdown_lines(self, summary: dict[str, Any]) -> list[str]:
        lines = [
            "[Timing Breakdown]",
            (
                "request_wall_time_ms = input_preprocess_time_ms + engine_pipeline_time_ms "
                f"= {self._format_ms(summary.get('input_preprocess_time_ms', 0.0))} + "
                f"{self._format_ms(summary.get('engine_pipeline_time_ms', 0.0))} "
                f"= {self._format_ms(summary.get('request_wall_time_ms', 0.0))}"
            ),
            (
                "engine_pipeline_time_ms = stage_gen_total_time_ms + "
                "stage_handoff_total_time_ms + final_output_overhead_time_ms "
                f"= {self._format_ms(summary.get('stage_gen_total_time_ms', 0.0))} + "
                f"{self._format_ms(summary.get('stage_handoff_total_time_ms', 0.0))} + "
                f"{self._format_ms(summary.get('final_output_overhead_time_ms', 0.0))} "
                f"= {self._format_ms(summary.get('engine_pipeline_time_ms', 0.0))}"
            ),
        ]

        ar2diffusion_total_ms = float(summary.get("ar2diffusion_total_time_ms", 0.0) or 0.0)
        if ar2diffusion_total_ms > 0.0:
            lines.append(
                "ar2diffusion_total_time_ms is included in stage_handoff_total_time_ms: "
                f"{self._format_ms(ar2diffusion_total_ms)} <= "
                f"{self._format_ms(summary.get('stage_handoff_total_time_ms', 0.0))}"
            )

        edge_handoff_fields = sorted(
            k
            for k in summary
            if k.startswith("stage_") and "_to_" in k and k.endswith("_handoff_time_ms")
        )
        for handoff_field in edge_handoff_fields:
            edge_name = handoff_field.removesuffix("_handoff_time_ms")
            ar2d_field = f"{edge_name}_ar2diffusion_time_ms"
            if ar2d_field in summary and float(summary.get(ar2d_field, 0.0) or 0.0) > 0.0:
                lines.append(
                    f"{ar2d_field} is included in {handoff_field}: "
                    f"{self._format_ms(summary[ar2d_field])} <= {self._format_ms(summary[handoff_field])}"
                )

        return lines

    def _get_or_create_transfer_event(
        self,
        from_stage: int,
        to_stage: int,
        request_id: str,
    ) -> TransferEdgeStats:
        key = (from_stage, to_stage, request_id)
        evt = self.transfer_events.get(key)
        if evt is None:
            evt = TransferEdgeStats(
                from_stage=from_stage,
                to_stage=to_stage,
                request_id=request_id,
                size_bytes=0,
                tx_time_ms=0.0,
                used_shm=False,
                rx_decode_time_ms=0.0,
                in_flight_time_ms=0.0,
            )
            self.transfer_events[key] = evt
        return evt

    def record_transfer_tx(
        self,
        from_stage: int,
        to_stage: int,
        request_id: Any,
        size_bytes: int,
        tx_time_ms: float,
        used_shm: bool,
    ) -> TransferEdgeStats | None:
        try:
            evt = self._get_or_create_transfer_event(
                int(from_stage),
                int(to_stage),
                str(request_id),
            )
            # Accumulate tx metrics
            evt.size_bytes += int(size_bytes)
            evt.tx_time_ms += float(tx_time_ms)
            evt.used_shm = evt.used_shm or bool(used_shm)
            return evt
        except Exception:
            return None

    def record_transfer_rx(
        self,
        stats: StageRequestStats,
    ) -> TransferEdgeStats | None:
        try:
            if stats.stage_id is None or stats.stage_id <= 0:
                return None
            from_stage = int(stats.stage_id) - 1
            to_stage = int(stats.stage_id)
            rid_key = str(stats.request_id)
            evt = self._get_or_create_transfer_event(from_stage, to_stage, rid_key)
            # Accumulate rx metrics
            if evt.size_bytes == 0:
                # size_bytes has been recorded in tx phase
                evt.size_bytes = int(stats.rx_transfer_bytes)
            evt.rx_decode_time_ms += float(stats.rx_decode_time_ms)
            evt.in_flight_time_ms += float(stats.rx_in_flight_time_ms)
            return evt
        except Exception:
            return None

    def record_audio_generated_frames(
        self,
        output_to_yield: Any,
        stage_id: int,
        request_id: str,
    ) -> None:
        try:
            if (
                output_to_yield.final_output_type == "audio"
                and (multimodal_output := output_to_yield.multimodal_output.get("audio")) is not None
                and len(multimodal_output) > 0
            ):
                nframes = sum(
                    int(t.shape[0]) if t.ndim > 0 else 1
                    for t in (multimodal_output if isinstance(multimodal_output, list) else [multimodal_output])
                )
                stage_events_for_req = self.stage_events.get(request_id, [])
                if stage_events_for_req:
                    for stage_event in stage_events_for_req:
                        if stage_event.stage_id == stage_id:
                            stage_event.audio_generated_frames += nframes
                            break
                else:
                    logger.warning(
                        "Failed to record audio generated frames for request %s at stage %s: no stage event found",
                        request_id,
                        stage_id,
                    )
        except Exception:
            logger.debug(
                "Failed to record audio frames for request %s",
                request_id,
                exc_info=True,
            )

    def process_stage_metrics(
        self,
        *,
        result: dict[str, Any],
        stage_type: str,
        stage_id: int,
        req_id: str,
        engine_outputs: Any,
        finished: bool,
        final_output_type: str | None,
        output_to_yield: Any | None,
    ) -> None:
        """Process and record stage metrics.

        Args:
            result: Result dict containing metrics from stage
            stage_type: Type of the stage (e.g., 'llm', 'diffusion')
            stage_id: Stage identifier
            req_id: Request identifier
            engine_outputs: Engine output object
            finished: Whether stage processing is finished
            final_output_type: Type of final output (e.g., 'text', 'audio')
            output_to_yield: Output object to attach metrics to
        """
        try:
            _m: StageRequestStats | None = result.get("metrics")

            # 1. Accumulate metrics from stage stats
            if _m is not None:
                self.accumulated_gen_time_ms[req_id][stage_id] += _m.stage_gen_time_ms
                self.accumulate_diffusion_metrics(stage_type, req_id, engine_outputs)
                if finished:
                    self.on_stage_metrics(stage_id, req_id, _m, final_output_type)

            # 2. No output to yield, nothing more to do
            if output_to_yield is None:
                return

            # 3. Not finished yet — empty metrics, skip audio recording
            if not finished:
                output_to_yield.metrics = {}
                return

            # 4. Finished with output: assign text metrics if available
            output_to_yield.metrics = {}
            stage_event = next(
                (evt for evt in reversed(self.stage_events.get(req_id, [])) if evt.stage_id == stage_id),
                None,
            )
            if stage_event is not None and stage_event.final_output_type == "text":
                output_to_yield.metrics = {
                    "num_tokens_in": stage_event.num_tokens_in,
                    "num_tokens_out": stage_event.num_tokens_out,
                    "stage_id": stage_event.stage_id,
                    "final_output_type": stage_event.final_output_type,
                }

            # 5. Finished: record audio generated frames
            self.record_audio_generated_frames(output_to_yield, stage_id, req_id)

        except Exception:
            logger.exception(
                "Failed to process metrics for stage %s, req %s",
                stage_id,
                req_id,
            )

    def _as_stage_request_stats(
        self,
        stage_id: int,
        req_id: str,
        metrics: StageRequestStats,
        final_output_type: str | None = None,
    ) -> StageRequestStats:
        "Convert dict to StageRequestStats if needed."
        stats = metrics
        stats.stage_id = stage_id
        stage_meta = self._get_stage_metadata(stage_id)
        stats.stage_type = self._get_stage_type(stage_meta)
        stats.stage_name = self._get_stage_name(stage_id, stage_meta)
        stats.request_id = req_id
        stats.final_output_type = final_output_type
        stats.diffusion_metrics = (
            {k: int(v) for k, v in self.diffusion_metrics.pop(req_id, {}).items()}
            if req_id in self.diffusion_metrics
            else None
        )
        return stats

    def on_stage_metrics(
        self,
        stage_id: int,
        req_id: Any,
        metrics: StageRequestStats,
        final_output_type: str | None = None,
    ) -> None:
        stats = self._as_stage_request_stats(stage_id, req_id, metrics, final_output_type)
        self.stage_total_tokens[stats.stage_id] += int(stats.num_tokens_out)
        if stats.stage_id == 0:
            self.stage_total_tokens[stats.stage_id] += int(stats.num_tokens_in)
        self.stage_events.setdefault(str(stats.request_id), []).append(stats)

        self.record_transfer_rx(stats)

    def _update_stage_event_field(self, stage_id: int, req_id: Any, field_name: str, value: float) -> None:
        rid_key = str(req_id)
        if rid_key in self.stage_events:
            for stats in self.stage_events[rid_key]:
                if stats.stage_id == stage_id:
                    setattr(stats, field_name, value if field_name == "handoff_to_stage_id" else float(value))
                    break
        else:
            logger.warning(
                "Failed to record %s for request %s at stage %s: no stage event found",
                field_name,
                req_id,
                stage_id,
            )

    def record_stage_postprocess_time(self, stage_id: int, req_id: Any, postproc_time_ms: float) -> None:
        self._update_stage_event_field(stage_id, req_id, "postprocess_time_ms", postproc_time_ms)

    def record_ar2diffusion_time(self, stage_id: int, req_id: Any, ar2diffusion_time_ms: float) -> None:
        self._update_stage_event_field(stage_id, req_id, "ar2diffusion_time_ms", ar2diffusion_time_ms)

    def record_stage_handoff_time(
        self,
        from_stage: int,
        to_stage: int,
        req_id: Any,
        handoff_time_ms: float,
    ) -> None:
        self._update_stage_event_field(from_stage, req_id, "handoff_to_stage_id", to_stage)
        self._update_stage_event_field(from_stage, req_id, "stage_handoff_time_ms", handoff_time_ms)

    @contextmanager
    def stage_postprocess_timer(self, stage_id: int, req_id: Any):
        """Context manager for measuring and recording stage postprocessing time.

        Usage:
            with metrics.stage_postprocess_timer(stage_id, request_id):
                next_inputs = next_stage.process_engine_inputs(...)
        """
        _t0 = time.perf_counter()
        try:
            yield
        finally:
            _postproc_ms = (time.perf_counter() - _t0) * 1000.0
            self.record_stage_postprocess_time(stage_id, req_id, _postproc_ms)

    def accumulate_diffusion_metrics(self, stage_type: str, req_id: Any, engine_outputs: Any) -> None:
        """Accumulate diffusion metrics for a request.

        Handles extraction and accumulation of diffusion stage metrics.

        Args:
            req_id: Request ID
            engine_outputs: Engine output object containing metrics
        """
        if stage_type != "diffusion":
            return
        engine_output = engine_outputs[0] if isinstance(engine_outputs, list) and engine_outputs else engine_outputs
        diffusion_metrics: dict = getattr(engine_output, "metrics", {})
        if isinstance(diffusion_metrics, list):
            diffusion_metrics = diffusion_metrics[0]
        if diffusion_metrics:
            for key, value in diffusion_metrics.items():
                self.diffusion_metrics[req_id][key] += value

    def on_forward(
        self,
        from_stage: int,
        to_stage: int,
        req_id: Any,
        size_bytes: int,
        tx_ms: float,
        used_shm: bool,
    ) -> None:
        # Mark first input time for the destination stage if not set
        if self.stage_first_ts[to_stage] is None:
            self.stage_first_ts[to_stage] = time.time()
        self.record_transfer_tx(
            from_stage=from_stage,
            to_stage=to_stage,
            request_id=req_id,
            size_bytes=size_bytes,
            tx_time_ms=tx_ms,
            used_shm=used_shm,
        )

    def on_finalize_request(
        self,
        stage_id: int,
        req_id: Any,
        req_start_ts: float,
        input_preprocess_time_ms: float = 0.0,
    ) -> None:
        rid_key = str(req_id)
        if rid_key in self.e2e_done:
            return  # Already finalized
        _t0 = float(req_start_ts)
        _t1 = time.time()
        # Update last output time for this stage
        prev_last = self.stage_last_ts[stage_id]
        self.stage_last_ts[stage_id] = _t1 if prev_last is None else max(prev_last, _t1)
        self.last_finish_ts = max(self.last_finish_ts, _t1)
        engine_pipeline_ms = (_t1 - _t0) * 1000.0
        input_preprocess_ms = float(input_preprocess_time_ms)
        request_wall_ms = engine_pipeline_ms + input_preprocess_ms

        # Sum tokens from all stages for this request
        # Include input tokens from stage 0 + output tokens from all stages
        total_tokens = 0
        if rid_key in self.stage_events:
            for evt in self.stage_events[rid_key]:
                if evt.stage_id == 0:
                    total_tokens += int(evt.num_tokens_in)
                total_tokens += int(evt.num_tokens_out)

        self.engine_pipeline_total_ms += engine_pipeline_ms
        self.input_preprocess_total_ms += input_preprocess_ms
        self.e2e_total_tokens += total_tokens
        self.e2e_count += 1
        self.e2e_done.add(rid_key)
        per_req_record = RequestE2EStats(
            request_id=rid_key,
            request_wall_time_ms=request_wall_ms,
            input_preprocess_time_ms=input_preprocess_ms,
            engine_pipeline_time_ms=engine_pipeline_ms,
            e2e_total_tokens=total_tokens,
            transfers_total_time_ms=float(
                sum(evt.total_time_ms for evt in self.transfer_events.values() if evt.request_id == rid_key)
            ),
            transfers_total_bytes=int(
                sum(evt.size_bytes for evt in self.transfer_events.values() if evt.request_id == rid_key)
            ),
        )
        self.e2e_events.append(per_req_record)

        if self.num_stages > 1:
            self._log_omni_timing(per_req_record)

    def build_and_log_summary(self) -> dict[str, Any]:
        if not self.log_stats:
            return {}
        wall_time_ms = max(0.0, (self.last_finish_ts - self.wall_start_ts) * 1000.0)
        e2e_avg_req = (wall_time_ms / self.e2e_count) if self.e2e_count > 0 else 0.0
        e2e_avg_tok = (self.e2e_total_tokens * 1000.0 / wall_time_ms) if wall_time_ms > 0 else 0.0

        if isinstance(self.final_stage_id_for_e2e, int):
            final_stage_id_map: dict[str, int] = {"*": int(self.final_stage_id_for_e2e)}
        else:
            final_stage_id_map = self.final_stage_id_for_e2e

        stage_wall_time_ms = [
            ((self.stage_last_ts[i] - self.stage_first_ts[i]) * 1000.0)
            if (self.stage_first_ts[i] is not None and self.stage_last_ts[i] is not None)
            else 0.0
            for i in range(self.num_stages)
        ]
        stage_gen_total_ms = 0.0
        stage_handoff_total_ms = 0.0
        ar2diffusion_total_ms = 0.0
        handoff_edge_totals: defaultdict[str, float] = defaultdict(float)
        handoff_edge_ar2diffusion: defaultdict[str, float] = defaultdict(float)
        for stage_evts in self.stage_events.values():
            for evt in stage_evts:
                stage_gen_total_ms += float(evt.stage_gen_time_ms or 0.0)
                handoff_ms = float(evt.stage_handoff_time_ms or 0.0)
                ar2d_ms = float(evt.ar2diffusion_time_ms or 0.0)
                stage_handoff_total_ms += handoff_ms
                ar2diffusion_total_ms += ar2d_ms
                if evt.stage_id is not None and evt.handoff_to_stage_id is not None:
                    edge = f"stage_{evt.stage_id}_to_{evt.handoff_to_stage_id}"
                    handoff_edge_totals[edge] += handoff_ms
                    handoff_edge_ar2diffusion[edge] += ar2d_ms
        final_output_overhead_ms = max(
            0.0,
            float(self.engine_pipeline_total_ms) - stage_gen_total_ms - stage_handoff_total_ms,
        )

        overall_summary = {
            "e2e_requests": int(self.e2e_count),
            "request_wall_time_ms": float(wall_time_ms),
            "input_preprocess_time_ms": float(self.input_preprocess_total_ms),
            "engine_pipeline_time_ms": float(self.engine_pipeline_total_ms),
            "stage_gen_total_time_ms": float(stage_gen_total_ms),
            "stage_handoff_total_time_ms": float(stage_handoff_total_ms),
            "ar2diffusion_total_time_ms": float(ar2diffusion_total_ms),
            "final_output_overhead_time_ms": float(final_output_overhead_ms),
            "e2e_total_tokens": int(self.e2e_total_tokens),
            "avg_request_wall_time_ms": float(e2e_avg_req),
            "avg_input_preprocess_time_ms": float(
                self.input_preprocess_total_ms / self.e2e_count if self.e2e_count > 0 else 0.0
            ),
            "avg_engine_pipeline_time_ms": float(
                self.engine_pipeline_total_ms / self.e2e_count if self.e2e_count > 0 else 0.0
            ),
            "avg_stage_gen_total_time_ms": float(stage_gen_total_ms / self.e2e_count if self.e2e_count > 0 else 0.0),
            "avg_stage_handoff_total_time_ms": float(
                stage_handoff_total_ms / self.e2e_count if self.e2e_count > 0 else 0.0
            ),
            "avg_ar2diffusion_time_ms": float(
                ar2diffusion_total_ms / self.e2e_count if self.e2e_count > 0 else 0.0
            ),
            "avg_final_output_overhead_time_ms": float(
                final_output_overhead_ms / self.e2e_count if self.e2e_count > 0 else 0.0
            ),
            "e2e_avg_tokens_per_s": float(e2e_avg_tok),
        }
        for edge, handoff_time in sorted(handoff_edge_totals.items()):
            overall_summary[f"{edge}_handoff_time_ms"] = float(handoff_time)
            ar2d_time = handoff_edge_ar2diffusion.get(edge, 0.0)
            if ar2d_time > 0.0:
                overall_summary[f"{edge}_ar2diffusion_time_ms"] = float(ar2d_time)
        # Add stage_wall_time_ms as separate fields for each stage
        for idx, wall_time in enumerate(stage_wall_time_ms):
            overall_summary[f"e2e_stage_{idx}_wall_time_ms"] = wall_time

        # Print overall summary
        # filter out all-zero fields for logging
        overall_fields = []
        avg_fields = {
            "avg_request_wall_time_ms",
            "avg_input_preprocess_time_ms",
            "avg_engine_pipeline_time_ms",
            "avg_stage_gen_total_time_ms",
            "avg_stage_handoff_total_time_ms",
            "avg_ar2diffusion_time_ms",
            "avg_final_output_overhead_time_ms",
        }
        dynamic_overall_fields = [
            k
            for k in overall_summary
            if k.startswith("stage_")
            and "_to_" in k
            and (k.endswith("_handoff_time_ms") or k.endswith("_ar2diffusion_time_ms"))
        ]
        stage_wall_fields = [k for k in overall_summary if k.startswith("e2e_stage_") and k.endswith("_wall_time_ms")]
        ordered_overall_fields = list(OVERALL_FIELDS or [])
        insert_at = ordered_overall_fields.index("e2e_total_tokens")
        ordered_overall_fields[insert_at:insert_at] = sorted(
            dynamic_overall_fields,
            key=lambda name: (name.replace("_ar2diffusion_time_ms", "_zz_ar2diffusion_time_ms")),
        ) + sorted(stage_wall_fields)
        for k in ordered_overall_fields or list(overall_summary.keys()):
            if self.e2e_count <= 1 and k in avg_fields:
                continue
            v = overall_summary.get(k, None)
            if v not in (0, 0.0, 0.000, None, ""):
                overall_fields.append(k)
        if overall_fields:
            logger.info(
                "\n%s",
                _format_table("Overall Summary", overall_summary, overall_fields),
            )
            logger.info("\n%s", "\n".join(self._build_timing_breakdown_lines(overall_summary)))

        all_request_ids = sorted(set(self.stage_events.keys()) | {e.request_id for e in self.e2e_events})

        result_stage_table = []
        result_trans_table = []
        result_e2e_table = []

        for rid in all_request_ids:
            # === E2E table (single column) ===
            e2e_evt = next((e for e in self.e2e_events if e.request_id == rid), None)
            if e2e_evt:
                e2e_data = _build_row(e2e_evt, E2E_FIELDS)
                result_e2e_table.append({"request_id": rid, **e2e_data})

            # === Stage table (columns = stage_id) ===
            stage_evts = sorted(
                self.stage_events.get(rid, []),
                key=lambda e: e.stage_id if e.stage_id is not None else -1,
            )
            # if any stage has diffusion_metrics, remove postprocess_time_ms field
            # because it is already included in diffusion_metrics
            local_exclude = STAGE_EXCLUDE.copy()
            has_diffusion_metrics = any(getattr(evt, "diffusion_metrics", None) for evt in stage_evts)
            if has_diffusion_metrics:
                local_exclude.add("postprocess_time_ms")
            local_stage_fields = _build_field_defs(StageRequestStats, local_exclude, FIELD_TRANSFORMS)

            # if diffusion_metrics is present, expand it into multiple columns
            # then remove diffusion_metrics from the table
            stage_rows = []
            for evt in stage_evts:
                row = {
                    "stage": self._stage_label(evt.stage_id),
                    "stage_id": evt.stage_id,
                    **_build_row(evt, local_stage_fields),
                }
                if evt.diffusion_metrics:
                    row.update(evt.diffusion_metrics)
                row.pop("diffusion_metrics", None)  # Remove the dict itself
                stage_rows.append(row)

            result_stage_table.append({"request_id": rid, "stages": stage_rows})

            if stage_rows:
                # filter out all-zero fields for logging
                all_value_fields = set()
                for row in stage_rows:
                    for k in row.keys():
                        if k not in ("stage", "stage_id"):
                            all_value_fields.add(k)
                value_fields_list = []
                for field in sorted(all_value_fields):
                    all_zero = True
                    for row in stage_rows:
                        v = row.get(field, None)
                        if v not in (0, 0.0, 0.000, None, ""):
                            all_zero = False
                            break
                    if not all_zero:
                        value_fields_list.append(field)

                if value_fields_list:
                    logger.info(
                        "\n%s",
                        _format_table(
                            f"StageRequestStats [request_id={rid}]",
                            stage_rows,
                            column_key="stage",
                            value_fields=value_fields_list,
                        ),
                    )

            # === Transfer table (columns = edge) ===
            transfer_evts = sorted(
                [e for e in self.transfer_events.values() if e.request_id == rid],
                key=lambda e: (e.from_stage, e.to_stage),
            )
            transfer_rows = [
                {"edge": f"{evt.from_stage}->{evt.to_stage}", **_build_row(evt, TRANSFER_FIELDS)}
                for evt in transfer_evts
            ]
            result_trans_table.append({"request_id": rid, "transfers": transfer_rows})

            if transfer_rows:
                # filter out all-zero fields for logging
                all_value_fields = set()
                for row in transfer_rows:
                    for k in row.keys():
                        if k != "edge":
                            all_value_fields.add(k)
                value_fields_list = []
                for field in sorted(all_value_fields):
                    all_zero = True
                    for row in transfer_rows:
                        v = row.get(field, None)
                        if v not in (0, 0.0, 0.000, None, ""):
                            all_zero = False
                            break
                    if not all_zero:
                        value_fields_list.append(field)

                if value_fields_list:
                    logger.info(
                        "\n%s",
                        _format_table(
                            f"TransferEdgeStats [request_id={rid}]",
                            transfer_rows,
                            column_key="edge",
                            value_fields=value_fields_list,
                        ),
                    )

        return {
            "final_stage_id": final_stage_id_map,
            "overall_summary": overall_summary,
            "stage_table": result_stage_table,
            "trans_table": result_trans_table,
            "e2e_table": result_e2e_table,
        }
