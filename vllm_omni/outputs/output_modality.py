# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Shared output modality type definitions."""

from __future__ import annotations

from typing import Literal, TypeAlias

FinalOutputModalityType: TypeAlias = Literal["text", "image", "audio", "video"]
