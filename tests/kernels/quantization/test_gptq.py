# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from tests.kernels.utils import opcheck
from vllm import _custom_ops as ops  # noqa: F401


def test_gptq_shuffle_opcheck():
    weight = torch.randint(
        -2000000, 2000000, (1792, 4096), device="cuda", dtype=torch.int32
    )
    perm = torch.empty((0,), device="cuda", dtype=torch.int32)
    bit = 4
    opcheck(torch.ops._C.gptq_shuffle, (weight, perm, bit))


def test_gptq_gemm_opcheck():
    a = torch.rand((240, 4096), device="cuda", dtype=torch.float16)
    weight = torch.randint(
        -2000000, 2000000, (512, 6144), device="cuda", dtype=torch.int32
    )
    zeros = torch.zeros((32, 768), device="cuda", dtype=torch.int32)
    scales = torch.rand((32, 6144), device="cuda", dtype=torch.float16)
    idx = torch.empty((0,), device="cuda", dtype=torch.int32)
    use_exllama = True
    bit = 4
    # Test both GPTQv1 and GPTQv2 format
    opcheck(
        torch.ops._C.gptq_gemm, (a, weight, zeros, scales, idx, use_exllama, True, bit)
    )
    opcheck(
        torch.ops._C.gptq_gemm, (a, weight, zeros, scales, idx, use_exllama, False, bit)
    )


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (6, 1),
    reason="Pascal DP4A kernel requires compute capability 6.1",
)
@pytest.mark.parametrize("use_v2_format", [False, True])
@pytest.mark.parametrize("size_k,group_size,atol", [(256, 64, 0.03), (4096, 128, 0.12)])
def test_gptq_gemm_pascal_dp4a_numerics(
    use_v2_format: bool, size_k: int, group_size: int, atol: float
):
    torch.manual_seed(0)
    size_n = 256
    groups = size_k // group_size

    q_values = torch.randint(0, 16, (size_k, size_n), device="cuda", dtype=torch.int64)
    shifts = torch.arange(0, 32, 4, device="cuda", dtype=torch.int64)
    qweight = (
        (q_values.reshape(size_k // 8, 8, size_n) << shifts[None, :, None])
        .sum(dim=1)
        .to(torch.int32)
    )

    zero_points = torch.randint(
        1, 16, (groups, size_n), device="cuda", dtype=torch.int64
    )
    stored_zeros = zero_points if use_v2_format else zero_points - 1
    qzeros = (
        (stored_zeros.reshape(groups, size_n // 8, 8) << shifts[None, None, :])
        .sum(dim=-1)
        .to(torch.int32)
    )
    scales = (
        torch.rand((groups, size_n), device="cuda", dtype=torch.float16) * 0.04 + 0.01
    )
    a = torch.randn((1, size_k), device="cuda", dtype=torch.float16) * 0.5
    empty_idx = torch.empty((0,), device="cuda", dtype=torch.int32)

    output = torch.ops._C.gptq_gemm(
        a, qweight, qzeros, scales, empty_idx, False, use_v2_format, 4
    )
    weight = (q_values - zero_points.repeat_interleave(group_size, dim=0)).float()
    weight *= scales.repeat_interleave(group_size, dim=0).float()
    reference = a.float() @ weight

    torch.testing.assert_close(output.float(), reference, rtol=0.03, atol=atol)
