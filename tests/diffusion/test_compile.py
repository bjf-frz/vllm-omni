# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch.nn as nn

import vllm_omni.diffusion.compile as compile_module
from vllm_omni.diffusion.compile import regionally_compile


class _WrappedBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.compile_called = False
        self.forward_compiled = False

    def compile(self, *args, **kwargs):
        self.compile_called = True
        return self

    def forward(self, x):
        return x


class _ModelWithWrappedRepeatedBlocks(nn.Module):
    _repeated_blocks = ["OriginalBlock"]
    _layerwise_offload_blocks_attrs = ["transformer_blocks"]

    def __init__(self) -> None:
        super().__init__()
        self.transformer_blocks = nn.ModuleList([_WrappedBlock(), _WrappedBlock()])
        self.other_blocks = nn.ModuleList([_WrappedBlock()])


def test_regionally_compile_matches_wrapped_blocks_by_declared_container_attr():
    model = _ModelWithWrappedRepeatedBlocks()

    regionally_compile(model)

    assert all(block.compile_called for block in model.transformer_blocks)
    assert not model.other_blocks[0].compile_called


def test_regionally_compile_can_compile_forward_instead_of_module_call(monkeypatch):
    model = _ModelWithWrappedRepeatedBlocks()
    compile_calls = []

    def _compile(fn, *args, **kwargs):
        compile_calls.append((fn, args, kwargs))

        def _compiled(*fn_args, **fn_kwargs):
            return fn(*fn_args, **fn_kwargs)

        return _compiled

    monkeypatch.setattr(compile_module.torch, "compile", _compile)

    regionally_compile(model, compile_forward=True, dynamic=True)

    assert len(compile_calls) == 2
    assert all(not block.compile_called for block in model.transformer_blocks)
    assert not model.other_blocks[0].compile_called
    assert model.transformer_blocks[0].forward("ok") == "ok"
