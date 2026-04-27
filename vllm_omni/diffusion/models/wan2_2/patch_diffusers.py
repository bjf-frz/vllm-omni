# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys

from vllm_omni.diffusion.layers.norm import RMSNormVAE


def patch_wan_rms_norm():
    """Patch diffusers Wan RMSNorm implementation with RMSNormVAE."""

    for module_name, module in list(sys.modules.items()):
        module_dict = getattr(module, "__dict__", None)
        if module_dict is not None and "WanRMS_norm" in module_dict:
            setattr(module, "WanRMS_norm", RMSNormVAE)
