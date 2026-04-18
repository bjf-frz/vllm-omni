#!/usr/bin/env python3
"""Micro-benchmark Wan AdaLayerNorm with a TensorRT-LLM graph.

This script intentionally stays outside the Wan2.2 PyTorch forward path. It is
for answering one question first: can TensorRT-LLM build and run the
``layer_norm(x) * (1 + scale) + shift`` pattern faster for Wan-like shapes?
It can also benchmark TensorRT-LLM's packaged ``AdaLayerNorm`` layer, which
includes the ``SiLU + Linear`` modulation projection inside the graph.

Example:
    python benchmarks/diffusion/wan_adalayernorm_trtllm_benchmark.py \
        --batch-size 1 --seq-len 32760 --hidden-size 5120 --dtype float16
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class BenchmarkResult:
    name: str
    median_ms: float
    mean_ms: float
    p90_ms: float
    min_ms: float
    max_ms: float
    iterations: int


def _numpy_dtype(dtype: str) -> str:
    if dtype == "bfloat16":
        raise ValueError("--trt-llm-impl layer_api currently supports float16/float32 weights only.")
    return {
        "float16": "float16",
        "float32": "float32",
    }[dtype]


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("This benchmark requires PyTorch with CUDA support.") from exc
    if not torch.cuda.is_available():
        raise SystemExit("This benchmark requires a CUDA device.")
    return torch


def _torch_dtype(torch: Any, dtype: str):
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping[dtype]


def _native_adalayernorm(torch: Any, x, scale, shift, eps: float):
    return torch.nn.functional.layer_norm(x.float(), (x.shape[-1],), None, None, eps) * (1 + scale) + shift


def _native_adalayernorm_layer_api(torch: Any, x, temb, weight, bias, eps: float):
    mod = torch.nn.functional.linear(torch.nn.functional.silu(temb), weight, bias)
    shift, scale = mod.chunk(2, dim=1)
    normed = torch.nn.functional.layer_norm(x.float(), (x.shape[-1],), None, None, eps)
    return normed * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def _summarize(name: str, samples_ms: list[float]) -> BenchmarkResult:
    samples = sorted(samples_ms)
    p90_index = min(len(samples) - 1, int(0.9 * (len(samples) - 1)))
    return BenchmarkResult(
        name=name,
        median_ms=statistics.median(samples),
        mean_ms=statistics.fmean(samples),
        p90_ms=samples[p90_index],
        min_ms=samples[0],
        max_ms=samples[-1],
        iterations=len(samples),
    )


def _time_cuda_callable(torch: Any, fn: Callable[[], Any], warmup: int, iterations: int) -> BenchmarkResult:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    samples_ms: list[float] = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(iterations):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples_ms.append(start.elapsed_time(end))
    return _summarize(fn.__name__, samples_ms)


def _build_trtllm_adalayernorm_session(
    *,
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    scale_seq_len: int,
    dtype: str,
    eps: float,
    impl: str,
    linear_weight=None,
    linear_bias=None,
):
    try:
        import tensorrt_llm
        from tensorrt_llm import Builder
        from tensorrt_llm.functional import Tensor, layer_norm
        try:
            from tensorrt_llm.layers import AdaLayerNorm
        except ImportError:
            from tensorrt_llm.layers.normalization import AdaLayerNorm
        from tensorrt_llm.network import net_guard
        from tensorrt_llm.runtime import Session
    except ImportError as exc:
        raise SystemExit(
            "TensorRT-LLM is not installed. Run this script in an NVIDIA TensorRT-LLM container "
            "or install a compatible tensorrt_llm package."
        ) from exc

    builder = Builder()
    network = builder.create_network()
    network.trt_network.name = "wan_adalayernorm"

    with net_guard(network):
        x = Tensor(name="x", dtype=dtype, shape=(batch_size, seq_len, hidden_size))
        if impl == "wan_functional":
            scale = Tensor(name="scale", dtype=dtype, shape=(batch_size, scale_seq_len, hidden_size))
            shift = Tensor(name="shift", dtype=dtype, shape=(batch_size, scale_seq_len, hidden_size))

            try:
                normalized = layer_norm(x, hidden_size, eps=eps)
            except TypeError:
                normalized = layer_norm(x, hidden_size, None, None, eps)
            y = normalized * (1 + scale) + shift
        elif impl == "layer_api":
            temb = Tensor(name="temb", dtype=dtype, shape=(batch_size, hidden_size))
            adaln = AdaLayerNorm(
                embedding_dim=hidden_size,
                output_dim=hidden_size * 2,
                norm_elementwise_affine=False,
                norm_eps=eps,
                chunk_dim=1,
                dtype=dtype,
            )
            if linear_weight is not None:
                adaln.linear.weight.value = linear_weight
            if linear_bias is not None and hasattr(adaln.linear, "bias"):
                adaln.linear.bias.value = linear_bias
            y = adaln(x, temb=temb)
        else:
            raise ValueError(f"Unsupported TensorRT-LLM AdaLayerNorm implementation: {impl}")
        y.mark_output("output", dtype)

    build_start = time.perf_counter()
    try:
        builder_config = builder.create_builder_config(name="wan_adalayernorm", precision=dtype)
    except TypeError:
        builder_config = builder.create_builder_config(precision=dtype)

    engine = builder.build_engine(network, builder_config)
    build_s = time.perf_counter() - build_start
    if engine is None:
        raise RuntimeError("TensorRT-LLM failed to build the Wan AdaLayerNorm engine.")

    session = Session.from_serialized_engine(engine)
    version = getattr(tensorrt_llm, "__version__", "unknown")
    return session, build_s, version


def _run_trtllm_benchmark(
    torch: Any,
    args: argparse.Namespace,
    x,
    inputs: dict[str, Any],
    expected_fn: Callable[[], Any],
    linear_weight=None,
    linear_bias=None,
) -> tuple[BenchmarkResult, float, str]:
    session, build_s, version = _build_trtllm_adalayernorm_session(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        hidden_size=args.hidden_size,
        scale_seq_len=inputs["scale"].shape[1] if "scale" in inputs else 1,
        dtype=args.dtype,
        eps=args.eps,
        impl=args.trt_llm_impl,
        linear_weight=linear_weight,
        linear_bias=linear_bias,
    )

    output = torch.empty_like(x)
    outputs = {"output": output}
    stream = torch.cuda.current_stream().cuda_stream

    session.set_shapes(inputs)

    def trt_llm_adalayernorm():
        ok = session.run(inputs=inputs, outputs=outputs, stream=stream)
        if not ok:
            raise RuntimeError("TensorRT-LLM session.run returned False.")
        return output

    if args.check_correctness:
        trt_llm_adalayernorm()
        torch.cuda.synchronize()
        expected = expected_fn().to(output.dtype)
        torch.testing.assert_close(output, expected, rtol=args.rtol, atol=args.atol)

    trt_llm_adalayernorm.__name__ = "trt_llm_adalayernorm"
    result = _time_cuda_callable(torch, trt_llm_adalayernorm, args.warmup, args.iterations)
    return result, build_s, version


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Wan AdaLayerNorm with TensorRT-LLM and PyTorch.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=32760)
    parser.add_argument("--hidden-size", type=int, default=5120)
    parser.add_argument(
        "--scale-shift-mode",
        choices=["broadcast", "token"],
        default="broadcast",
        help="broadcast uses [B,1,D] scale/shift; token uses [B,S,D].",
    )
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--skip-trt-llm", action="store_true", help="Only run the PyTorch baseline.")
    parser.add_argument(
        "--trt-llm-impl",
        choices=["wan_functional", "layer_api"],
        default="wan_functional",
        help=(
            "wan_functional exactly benchmarks Wan's external scale/shift pattern. "
            "layer_api uses TensorRT-LLM's packaged AdaLayerNorm layer, including SiLU+Linear."
        ),
    )
    parser.add_argument("--check-correctness", action="store_true", help="Compare TensorRT-LLM output to PyTorch.")
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    torch = _require_torch()
    dtype = _torch_dtype(torch, args.dtype)
    device = torch.device("cuda")
    scale_seq_len = 1 if args.scale_shift_mode == "broadcast" else args.seq_len

    torch.manual_seed(0)
    x = torch.randn(args.batch_size, args.seq_len, args.hidden_size, device=device, dtype=dtype)
    linear_weight = None
    linear_bias = None

    if args.trt_llm_impl == "wan_functional":
        scale = torch.randn(args.batch_size, scale_seq_len, args.hidden_size, device=device, dtype=dtype)
        shift = torch.randn(args.batch_size, scale_seq_len, args.hidden_size, device=device, dtype=dtype)
        inputs = {
            "x": x,
            "scale": scale,
            "shift": shift,
        }

        def pytorch_adalayernorm():
            return _native_adalayernorm(torch, x, scale, shift, args.eps)

        shape_payload = {
            "x": list(x.shape),
            "scale": list(scale.shape),
            "shift": list(shift.shape),
        }
    else:
        if args.scale_shift_mode != "broadcast":
            raise SystemExit("TensorRT-LLM AdaLayerNorm layer_api only supports broadcast modulation in this script.")
        temb = torch.randn(args.batch_size, args.hidden_size, device=device, dtype=dtype)
        weight = torch.randn(args.hidden_size * 2, args.hidden_size, device=device, dtype=dtype) * 0.01
        bias = torch.randn(args.hidden_size * 2, device=device, dtype=dtype) * 0.01
        linear_weight = weight.detach().cpu().numpy().astype(_numpy_dtype(args.dtype))
        linear_bias = bias.detach().cpu().numpy().astype(_numpy_dtype(args.dtype))
        inputs = {
            "x": x,
            "temb": temb,
        }

        def pytorch_adalayernorm():
            return _native_adalayernorm_layer_api(torch, x, temb, weight, bias, args.eps)

        shape_payload = {
            "x": list(x.shape),
            "temb": list(temb.shape),
            "linear_weight": list(weight.shape),
            "linear_bias": list(bias.shape),
        }

    pytorch_adalayernorm.__name__ = "pytorch_adalayernorm"
    results: list[BenchmarkResult] = [
        _time_cuda_callable(torch, pytorch_adalayernorm, args.warmup, args.iterations)
    ]

    trt_build_s = None
    trtllm_version = None
    if not args.skip_trt_llm:
        trt_result, trt_build_s, trtllm_version = _run_trtllm_benchmark(
            torch,
            args,
            x,
            inputs,
            pytorch_adalayernorm,
            linear_weight=linear_weight,
            linear_bias=linear_bias,
        )
        results.append(trt_result)

    payload: dict[str, Any] = {
        "shape": shape_payload,
        "dtype": args.dtype,
        "eps": args.eps,
        "trt_llm_impl": args.trt_llm_impl,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "check_correctness": args.check_correctness,
        "trt_llm_build_s": trt_build_s,
        "trt_llm_version": trtllm_version,
        "results": [asdict(result) for result in results],
    }

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"Shape: {shape_payload}")
        print(f"TensorRT-LLM implementation: {args.trt_llm_impl}")
        print(f"dtype={args.dtype}, eps={args.eps}, warmup={args.warmup}, iterations={args.iterations}")
        if trt_build_s is not None:
            print(f"TensorRT-LLM build: {trt_build_s:.3f}s (version={trtllm_version})")
        for result in results:
            print(
                f"{result.name}: median={result.median_ms:.4f} ms, mean={result.mean_ms:.4f} ms, "
                f"p90={result.p90_ms:.4f} ms, min={result.min_ms:.4f} ms, max={result.max_ms:.4f} ms"
            )


if __name__ == "__main__":
    main()
