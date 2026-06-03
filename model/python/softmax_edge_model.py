#!/usr/bin/env python3

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import random
import struct
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover
    np = None  # type: ignore[assignment]

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


DEFAULT_TOPK_VALUES: Tuple[int, ...] = (1, 3, 5)
DEFAULT_COMPARE_PROFILES: Tuple[str, ...] = ("doc_adaptive_desc8_q6_special4", "custom")


def special_descriptor_frac_width(descriptor_mode: str) -> int | None:
    if descriptor_mode == "z_code_q6_special_4":
        return 6
    if descriptor_mode == "z_code_q7_special_4":
        return 7
    return None


@dataclass(frozen=True)
class FixedFormat:

    int_width: int
    frac_width: int
    signed: bool = True

    @property
    def total_width(self) -> int:
        return self.int_width + self.frac_width

    @property
    def scale(self) -> int:
        return 1 << self.frac_width

    @property
    def min_raw(self) -> int:
        if not self.signed:
            return 0
        return -(1 << (self.total_width - 1))

    @property
    def max_raw(self) -> int:
        if not self.signed:
            return (1 << self.total_width) - 1
        return (1 << (self.total_width - 1)) - 1

    def quantize(self, value: float) -> int:
        if math.isnan(value):
            value = 0.0
        if math.isinf(value):
            return self.max_raw if value > 0 else self.min_raw
        raw = int(round(value * self.scale))
        return max(self.min_raw, min(self.max_raw, raw))

    def to_float(self, raw: int) -> float:
        return raw / self.scale


@dataclass(frozen=True)
class DescriptorConfig:

    shift_width: int
    frac_width: int


@dataclass(frozen=True)
class SoftmaxModelConfig:

    input_format: str = "fp16"
    bus_width: int | None = None
    elem_width: int = 16
    lane_count: int = 8
    row_depth: int = 256
    block_size: int | None = None
    fx_int_width: int = 6
    fx_frac_width: int = 10
    y_int_width: int = 6
    y_frac_width: int = 10
    exp_frac_width: int = 10
    out_frac_width: int = 10
    sum_int_width: int = 10
    sum_frac_width: int = 10
    y_shift_width: int = 4
    y_frac_buf_width: int = 10
    descriptor_mode: str = "shift_frac"
    sum_use_buffered_descriptor: bool = False
    prune_threshold: float = -10.0
    adaptive_prune_mode: str = "fixed"
    adaptive_c2_threshold: int = 3
    adaptive_tau_aggressive: int = -2
    adaptive_tau_conservative: int = -4
    elem_prune_compare_mode: str = "floor_int"
    block_prune_enabled: bool = True
    block_prune_threshold: float | None = None
    exp_pwl_mode: str = "DOC"
    rtl_exact: bool = False
    log2e_shift_terms: Tuple[Tuple[int, int], ...] = (
        (1, 0),
        (1, 1),
        (-1, 4),
    )

    @property
    def lane_count_effective(self) -> int:
        if self.lane_count <= 0:
            raise ValueError("lane_count must be positive")
        return self.lane_count

    @property
    def lane_num(self) -> int:
        return self.lane_count_effective

    @property
    def row_len(self) -> int:
        return self.row_depth

    @property
    def bus_width_effective(self) -> int:
        if self.bus_width is None:
            return self.lane_count_effective * self.elem_width
        if self.bus_width <= 0:
            raise ValueError("bus_width must be positive")
        return self.bus_width

    @property
    def block_size_effective(self) -> int:
        if self.block_size is None:
            return self.lane_count_effective
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")
        return self.block_size

    @property
    def block_clog2(self) -> int:
        return (self.block_size_effective - 1).bit_length()

    @property
    def fx_fmt(self) -> FixedFormat:
        return FixedFormat(self.fx_int_width, self.fx_frac_width, signed=True)

    @property
    def y_fmt(self) -> FixedFormat:
        return FixedFormat(self.y_int_width, self.y_frac_width, signed=True)

    @property
    def exp_fmt(self) -> FixedFormat:
        return FixedFormat(1, self.exp_frac_width, signed=False)

    @property
    def sum_fmt(self) -> FixedFormat:
        return FixedFormat(self.sum_int_width, self.sum_frac_width, signed=False)

    @property
    def out_fmt(self) -> FixedFormat:
        return FixedFormat(1, self.out_frac_width, signed=False)

    @property
    def desc_cfg(self) -> DescriptorConfig:
        return DescriptorConfig(self.y_shift_width, self.y_frac_buf_width)

    @property
    def block_prune_threshold_effective(self) -> float:
        if self.block_prune_threshold is not None:
            return float(self.block_prune_threshold)
        # A whole block can be skipped once its aligned sum_local would quantize below
        # one LSB of the running denominator. That bound depends on the denominator
        # fractional precision and the block's worst-case accumulation span.
        return float(-(self.sum_frac_width + self.block_clog2))

    @property
    def prune_threshold_is_integral(self) -> bool:
        return math.isfinite(self.prune_threshold) and math.isclose(
            self.prune_threshold, round(self.prune_threshold), abs_tol=1e-12
        )

    @property
    def prune_threshold_int(self) -> int:
        if not self.prune_threshold_is_integral:
            raise ValueError("integer-part element prune modes require an integer prune_threshold")
        return int(round(self.prune_threshold))

    @property
    def adaptive_prune_enabled(self) -> bool:
        return self.adaptive_prune_mode != "fixed"

    @property
    def adaptive_tau_aggressive_int(self) -> int:
        return int(self.adaptive_tau_aggressive)

    @property
    def adaptive_tau_conservative_int(self) -> int:
        return int(self.adaptive_tau_conservative)


@dataclass
class SoftmaxDescriptor:

    y_shift: int
    y_frac_raw: int
    y_prune: bool


@dataclass
class BlockMeta:

    row_id: int
    block_index: int
    block_max_value: int
    block_pruned: bool
    block_sum_raw: int
    descriptor_start_index: int
    element_count: int
    partial_valid: bool
    complete: bool
    tau_elem_value: float = 0.0
    tau_blk_value: float = 0.0
    c2_count: int = 0


@dataclass
class RowFinalStateToken:

    row_id: int
    row_bank: int
    row_max_value_final: int
    row_sum_final_raw: int
    block_count: int


@dataclass
class RowNormalizationToken:

    row_id: int
    row_bank: int
    row_max_value_final: int
    normalized_row_sum_exponent: int
    denom_k_raw: int
    denom_k: float
    denom_delta_raw: int
    denom_delta: float
    block_count: int


@dataclass
class SoftmaxRowResult:

    input_row: List[float]
    fp_quantized_row: List[float]
    fx_raw_row: List[int]
    fx_row: List[float]
    block_max_integer_parts: List[int]
    block_max_values: List[int]
    block_prune_deltas: List[float]
    row_max_value_final: int
    block_prune_flags: List[bool]
    y_raw_row: List[int]
    y_row: List[float]
    element_prune_flags: List[bool]
    prune_flags: List[bool]
    descriptors: List[SoftmaxDescriptor]
    block_metas: List[BlockMeta]
    block_sum_raw_values: List[int]
    row_sum_final_raw: int
    row_sum_final: float
    row_final_state_token: RowFinalStateToken
    row_normalization_token: RowNormalizationToken
    normalized_row_sum_mantissa_raw: int
    normalized_row_sum_mantissa: float
    normalized_row_sum_exponent: int
    denom_k_raw: int
    denom_k: float
    denom_delta_raw: int
    denom_delta: float
    approx_probs: List[float]
    approx_probs_raw: List[int]
    reference_probs: List[float]
    max_abs_error: float
    mean_abs_error: float
    mse: float
    kl_divergence: float

    @property
    def row_ctx(self) -> RowFinalStateToken:
        return self.row_final_state_token

    @property
    def module6_denom_state(self) -> RowNormalizationToken:
        return self.row_normalization_token


@dataclass
class SoftmaxBatchEvalSummary:

    profile: str
    num_rows: int
    row_depth: int
    seed: int
    value_min: float
    value_max: float
    topk_values: List[int]
    topk_match_counts: Dict[str, int]
    topk_match_rates: Dict[str, float]
    top1_match_count: int
    top1_match_rate: float
    mae: float
    mse: float
    avg_max_abs_error: float
    avg_mean_abs_error: float
    global_max_abs_error: float
    avg_l1_error: float
    avg_prob_sum: float
    avg_prob_sum_error: float
    block_pruned_block_count: int
    block_pruned_element_count: int
    element_pruned_only_count: int
    total_pruned_element_count: int
    avg_kl_divergence: float
    avg_perplexity: float


@dataclass(frozen=True)
class WorkloadSpec:

    name: str
    default_row_depth: int
    distribution: str
    value_min: float
    value_max: float
    normal_mean: float
    normal_std: float


def quantize_fp16(value: float) -> float:
    try:
        return struct.unpack(">e", struct.pack(">e", float(value)))[0]
    except OverflowError:
        return float("inf") if value > 0 else float("-inf")


def float_to_fp16_bits(value: float) -> int:
    """Pack a Python float into IEEE FP16 and return the raw 16-bit pattern."""
    return int.from_bytes(struct.pack(">e", float(value)), byteorder="big")


def quantize_bf16(value: float) -> float:
    """Round a Python float to BF16 and back.

    BF16 keeps the 8-bit exponent of FP32 and truncates most mantissa bits.
    """
    if math.isnan(value):
        return float("nan")
    packed = struct.pack(">f", float(value))
    bits = struct.unpack(">I", packed)[0]
    round_bias = 0x7FFF + ((bits >> 16) & 1)
    rounded = (bits + round_bias) & 0xFFFFFFFF
    bf16_bits = rounded & 0xFFFF0000
    return struct.unpack(">f", struct.pack(">I", bf16_bits))[0]


def quantize_input_format(value: float, input_format: str) -> float:
    if input_format == "fp16":
        return quantize_fp16(value)
    if input_format == "bf16":
        return quantize_bf16(value)
    raise ValueError(f"unsupported input format: {input_format}")


def split_blocks(values: Sequence[int], block_size: int) -> List[List[int]]:
    return [list(values[idx : idx + block_size]) for idx in range(0, len(values), block_size)]


def parse_topk_values(topk_text: str) -> Tuple[int, ...]:
    """Parse top-k list like '1,3,5' and always keep k=1."""
    parsed = {1}
    for token in topk_text.split(","):
        token = token.strip()
        if not token:
            continue
        value = int(token)
        if value <= 0:
            raise ValueError("top-k values must be positive integers")
        parsed.add(value)
    return tuple(sorted(parsed))


def topk_indices(values: Sequence[float], k: int) -> Tuple[int, ...]:
    """Return top-k indices in descending score order, tie-breaking by index."""
    limit = min(k, len(values))
    order = sorted(range(len(values)), key=lambda idx: (-values[idx], idx))
    return tuple(order[:limit])


def fixed_floor_int(raw_value: int, frac_width: int) -> int:
    return raw_value >> frac_width


def fixed_trunc_int(raw_value: int, frac_width: int) -> int:
    if raw_value >= 0:
        return raw_value >> frac_width
    return -((-raw_value) >> frac_width)


def fixed_ceil_int(raw_value: int, frac_width: int) -> int:
    floor_int = fixed_floor_int(raw_value, frac_width)
    frac_mask = (1 << frac_width) - 1
    if raw_value & frac_mask:
        return floor_int + 1
    return floor_int


def rescale_signed_raw(raw_value: int, src_frac_width: int, dst_frac_width: int) -> int:
    """Shift signed fixed-point raw values between Q domains with RTL-style truncation."""
    if dst_frac_width == src_frac_width:
        return raw_value
    if dst_frac_width > src_frac_width:
        return raw_value << (dst_frac_width - src_frac_width)
    return raw_value >> (src_frac_width - dst_frac_width)


def rescale_frac_raw(raw_value: int, src_frac_width: int, dst_frac_width: int) -> int:
    if dst_frac_width == src_frac_width:
        return raw_value
    if dst_frac_width > src_frac_width:
        return raw_value << (dst_frac_width - src_frac_width)
    shift = src_frac_width - dst_frac_width
    return (raw_value + (1 << (shift - 1))) >> shift


def clamp_raw(raw_value: int, fmt: FixedFormat) -> int:
    return max(fmt.min_raw, min(fmt.max_raw, raw_value))


def fp16_bits_to_q6_10_raw(fp16_bits: int) -> int:
    """Mirror ``softmax_m1_preprocess.fp16_to_q6_10`` exactly."""
    sign_bit = (fp16_bits >> 15) & 1
    exp_field = (fp16_bits >> 10) & 0x1F
    frac_field = fp16_bits & 0x3FF
    if exp_field == 0:
        return 0
    if exp_field == 0x1F:
        return -32768 if sign_bit else 32767

    exp_unbias = exp_field - 15
    scaled_raw = (1 << 10) | frac_field
    if exp_unbias >= 0:
        scaled_raw <<= exp_unbias
    else:
        scaled_raw >>= -exp_unbias
    if sign_bit:
        scaled_raw = -scaled_raw
    return max(-32768, min(32767, scaled_raw))


def rtl_m1_fp16_to_base2_q6_10(fp16_bits: int) -> int:
    """Mirror ``softmax_m1_preprocess`` exactly, including the shift-add log2(e)."""
    q6_10_raw = fp16_bits_to_q6_10_raw(fp16_bits)
    base2_raw = q6_10_raw + (q6_10_raw >> 1) - (q6_10_raw >> 4)
    return max(-32768, min(32767, base2_raw))


def shift_add_multiply(raw_value: int, shift_terms: Iterable[Tuple[int, int]]) -> int:
    result = 0
    for sign, shift in shift_terms:
        term = raw_value if shift == 0 else (raw_value >> shift)
        result += sign * term
    return result


def approx_exp2_frac(frac_value: float, mode: str) -> float:
    if not 0.0 <= frac_value < 1.0:
        frac_value = min(max(frac_value, 0.0), math.nextafter(1.0, 0.0))
    if mode == "DOC":
        folded = frac_value if frac_value < 0.5 else (1.0 - frac_value)
        return 1.0 + frac_value - 0.1875 * folded
    raise ValueError(f"unsupported exp PWL mode: {mode}")


def approx_exp2_frac_raw(frac_raw: int, frac_width: int, mode: str) -> int:
    unit_raw = 1 << frac_width
    frac_raw = max(0, min(unit_raw - 1, frac_raw))
    if mode == "DOC":
        tri_raw = frac_raw if frac_raw < (unit_raw >> 1) else (unit_raw - frac_raw)
        corr_raw = (tri_raw >> 3) + (tri_raw >> 4)
        return unit_raw + frac_raw - corr_raw
    raise ValueError(f"unsupported exp PWL mode: {mode}")


def approx_delta_triangle(k_value: float) -> float:
    if not 0.0 <= k_value < 1.0:
        k_value = min(max(k_value, 0.0), math.nextafter(1.0, 0.0))
    folded = k_value if k_value < 0.5 else (1.0 - k_value)
    return 0.1875 * folded


def approx_delta_triangle_raw(k_raw: int, frac_width: int) -> int:
    unit_raw = 1 << frac_width
    k_raw = max(0, min(unit_raw - 1, k_raw))
    folded_raw = k_raw if k_raw < (unit_raw >> 1) else (unit_raw - k_raw)
    return (folded_raw * 3 + 8) >> 4


def normalize_sum(sum_raw: int, frac_width: int) -> Tuple[int, int]:
    if sum_raw <= 0:
        return 0, 0
    highest_bit = sum_raw.bit_length() - 1
    exponent = highest_bit - frac_width
    if exponent >= 0:
        mant_raw = sum_raw >> exponent
    else:
        mant_raw = sum_raw << (-exponent)
    return mant_raw, exponent


def quantize_output_probability(prob_value: float, frac_width: int) -> int:
    scale = 1 << frac_width
    if prob_value <= 0.0:
        return 0
    if prob_value >= 1.0:
        return scale
    return int(math.floor(prob_value * scale + 0.5))


def exact_softmax(row: Sequence[float]) -> List[float]:
    if not row:
        return []
    max_value = max(row)
    exp_values = [math.exp(value - max_value) for value in row]
    denom = sum(exp_values)
    if denom == 0.0:
        return [0.0 for _ in row]
    return [value / denom for value in exp_values]


def calculate_kl_divergence(p: Sequence[float], q: Sequence[float], eps: float = 1e-10) -> float:
    kl = 0.0
    for p_val, q_val in zip(p, q):
        if p_val > 0:
            kl += p_val * math.log((p_val + eps) / (q_val + eps))
    return max(0.0, kl)


class SoftmaxEdgeModel:

    def __init__(self, cfg: SoftmaxModelConfig):
        self.cfg = cfg
        self._fp16_to_fx_lut_np: np.ndarray | None = None
        self._exp_lut_np: Dict[int, np.ndarray] = {}
        self._delta_lut_np: Dict[int, np.ndarray] = {}
        self._validate_descriptor_cfg()
        if self.cfg.rtl_exact:
            self._validate_rtl_exact_cfg()
        self._init_fast_luts()

    def _init_fast_luts(self) -> None:
        if np is not None:
            for frac_width in (5, 6):
                unit_raw = 1 << frac_width
                self._exp_lut_np[frac_width] = np.asarray(
                    [approx_exp2_frac_raw(frac_raw, frac_width, "DOC") for frac_raw in range(unit_raw)],
                    dtype=np.int32,
                )
                self._delta_lut_np[frac_width] = np.asarray(
                    [approx_delta_triangle_raw(k_raw, frac_width) for k_raw in range(unit_raw)],
                    dtype=np.int32,
                )
            if self.rtl_exact_enabled() and self.cfg.input_format == "fp16":
                self._fp16_to_fx_lut_np = np.asarray(
                    [rtl_m1_fp16_to_base2_q6_10(bits) for bits in range(1 << 16)],
                    dtype=np.int32,
                )

    def _get_exp_lut_np(self, frac_width: int) -> "np.ndarray":
        if np is None:
            raise RuntimeError("NumPy fast LUT requested without NumPy support")
        lut = self._exp_lut_np.get(frac_width)
        if lut is None:
            unit_raw = 1 << frac_width
            lut = np.asarray(
                [approx_exp2_frac_raw(frac_raw, frac_width, self.cfg.exp_pwl_mode) for frac_raw in range(unit_raw)],
                dtype=np.int32,
            )
            self._exp_lut_np[frac_width] = lut
        return lut

    def _get_delta_lut_np(self, frac_width: int) -> "np.ndarray":
        if np is None:
            raise RuntimeError("NumPy fast LUT requested without NumPy support")
        lut = self._delta_lut_np.get(frac_width)
        if lut is None:
            unit_raw = 1 << frac_width
            lut = np.asarray(
                [approx_delta_triangle_raw(k_raw, frac_width) for k_raw in range(unit_raw)],
                dtype=np.int32,
            )
            self._delta_lut_np[frac_width] = lut
        return lut

    def _validate_descriptor_cfg(self) -> None:
        supported_modes = {"shift_frac", "z_code_q6_special_4", "z_code_q7_special_4"}
        if self.cfg.descriptor_mode not in supported_modes:
            raise ValueError(
                f"unsupported descriptor_mode: {self.cfg.descriptor_mode} "
                f"(expected one of {sorted(supported_modes)!r})"
            )
        special_frac_width = special_descriptor_frac_width(self.cfg.descriptor_mode)
        if special_frac_width is None:
            return

        mismatches: List[str] = []
        if self.cfg.y_frac_width != special_frac_width:
            mismatches.append(f"y_frac_width={self.cfg.y_frac_width!r}")
        if self.cfg.y_frac_buf_width != special_frac_width:
            mismatches.append(f"y_frac_buf_width={self.cfg.y_frac_buf_width!r}")
        descriptor_width = self.cfg.y_shift_width + self.cfg.y_frac_buf_width
        if descriptor_width != (self.cfg.y_shift_width + special_frac_width):
            mismatches.append(
                f"descriptor_width={descriptor_width!r}"
            )
        if not self.cfg.sum_use_buffered_descriptor:
            mismatches.append(f"sum_use_buffered_descriptor={self.cfg.sum_use_buffered_descriptor!r}")
        if mismatches:
            raise ValueError(
                f"descriptor_mode={self.cfg.descriptor_mode!r} requires q{special_frac_width} buffered descriptors "
                f"in {self.cfg.y_shift_width + special_frac_width} bits: "
                + ", ".join(mismatches)
            )

    def _build_live_descriptor(self, y_raw: int) -> SoftmaxDescriptor:
        special_frac_width = special_descriptor_frac_width(self.cfg.descriptor_mode)
        if special_frac_width is not None:
            unit_raw = 1 << self.cfg.y_frac_width
            z_mag_raw = max(0, -y_raw)
            if z_mag_raw == 0:
                return SoftmaxDescriptor(y_shift=0, y_frac_raw=0, y_prune=False)
            # Saturate the top bin: any surviving value in [3.984375, 4.0]
            # collapses to the largest finite q-format descriptor, avoiding a
            # dedicated special-code branch in the accuracy experiment.
            if z_mag_raw >= (4 * unit_raw) - 1:
                return SoftmaxDescriptor(y_shift=4, y_frac_raw=1, y_prune=False)
            y_shift = (z_mag_raw + unit_raw - 1) >> self.cfg.y_frac_width
            frac_raw = (y_shift << self.cfg.y_frac_width) - z_mag_raw
            return SoftmaxDescriptor(y_shift=y_shift, y_frac_raw=frac_raw, y_prune=False)

        if self.rtl_exact_enabled():
            if y_raw == 0:
                return SoftmaxDescriptor(y_shift=0, y_frac_raw=0, y_prune=False)
            y_shift = ((-y_raw) + ((1 << self.cfg.y_frac_width) - 1)) >> self.cfg.y_frac_width
            frac_raw = y_raw + (y_shift << self.cfg.y_frac_width)
            return SoftmaxDescriptor(
                y_shift=y_shift & ((1 << self.cfg.y_shift_width) - 1),
                y_frac_raw=frac_raw & ((1 << self.cfg.y_frac_buf_width) - 1),
                y_prune=False,
            )

        y_floor = fixed_floor_int(y_raw, self.cfg.y_frac_width)
        frac_raw = y_raw - (y_floor << self.cfg.y_frac_width)
        y_shift = -y_floor
        buf_frac_raw = rescale_frac_raw(frac_raw, self.cfg.y_frac_width, self.cfg.y_frac_buf_width)
        return SoftmaxDescriptor(
            y_shift=max(0, min((1 << self.cfg.y_shift_width) - 1, y_shift)),
            y_frac_raw=buf_frac_raw,
            y_prune=False,
        )

    def _validate_rtl_exact_cfg(self) -> None:
        special_frac_width = special_descriptor_frac_width(self.cfg.descriptor_mode)
        if special_frac_width is not None:
            expected_scalars = (
                ("input_format", "fp16"),
                ("fx_int_width", 6),
                ("fx_frac_width", 10),
                ("y_int_width", 3),
                ("y_frac_width", special_frac_width),
                ("exp_frac_width", special_frac_width),
                ("out_frac_width", 15),
                ("sum_int_width", 10),
                ("sum_frac_width", special_frac_width),
                ("y_shift_width", 2),
                ("y_frac_buf_width", special_frac_width),
                ("descriptor_mode", self.cfg.descriptor_mode),
                ("sum_use_buffered_descriptor", True),
                ("elem_prune_compare_mode", "floor_int"),
                ("exp_pwl_mode", "DOC"),
            )
        mismatches: List[str] = []
        for field_name, expected_value in expected_scalars:
            if getattr(self.cfg, field_name) != expected_value:
                mismatches.append(f"{field_name}={getattr(self.cfg, field_name)!r}")
        if self.cfg.log2e_shift_terms != ((1, 0), (1, 1), (-1, 4)):
            mismatches.append(f"log2e_shift_terms={self.cfg.log2e_shift_terms!r}")
        if mismatches:
            raise ValueError(
                "rtl_exact=True currently only supports the shipped special-descriptor configs: "
                + ", ".join(mismatches)
            )

    def rtl_exact_enabled(self) -> bool:
        return self.cfg.rtl_exact

    def block_prune_threshold_value(self) -> float:
        return self.cfg.block_prune_threshold_effective

    def element_prune_integer_extract(self, raw_value: int, compare_mode: str) -> int:
        if compare_mode == "floor_int":
            return fixed_floor_int(raw_value, self.cfg.fx_frac_width)
        if compare_mode == "trunc_int":
            return fixed_trunc_int(raw_value, self.cfg.fx_frac_width)
        raise ValueError(f"integer extract is unsupported for compare mode: {compare_mode}")

    def resolve_block_prune_policy(
        self,
        block: Sequence[int],
        block_max_value: int,
        prune_compare_mode: str,
    ) -> Tuple[float, int, float, int]:
        if self.cfg.adaptive_prune_mode == "fixed":
            tau_elem = float(self.cfg.prune_threshold)
            tau_elem_int = self.cfg.prune_threshold_int if prune_compare_mode != "full_y" else 0
            tau_blk = self.block_prune_threshold_value()
            return tau_elem, tau_elem_int, tau_blk, 0

        if self.cfg.adaptive_prune_mode != "block_c2_two_level":
            raise ValueError(f"unsupported adaptive_prune_mode: {self.cfg.adaptive_prune_mode}")
        if prune_compare_mode == "full_y":
            raise ValueError("adaptive block-local prune mode does not support full_y compare mode")

        c2_count = sum(
            1
            for fx_raw in block
            if self.element_prune_integer_extract(fx_raw, prune_compare_mode) >= block_max_value - 2
        )
        if c2_count <= self.cfg.adaptive_c2_threshold:
            tau_elem_int = self.cfg.adaptive_tau_aggressive_int
        else:
            tau_elem_int = self.cfg.adaptive_tau_conservative_int
        tau_elem = float(tau_elem_int)
        if self.cfg.block_prune_threshold is not None:
            tau_blk = float(self.cfg.block_prune_threshold)
        else:
            tau_blk = self.block_prune_threshold_value()
        return tau_elem, tau_elem_int, tau_blk, c2_count

    def module1_fp_to_fx(self, row: Sequence[float]) -> Tuple[List[float], List[int], List[float]]:
        if self.rtl_exact_enabled() and self.cfg.input_format == "fp16" and np is not None:
            row_np = np.asarray(row, dtype=np.float32)
            with np.errstate(over="ignore"):
                fp16_row_np = row_np.astype(np.float16)
            fp_quantized = fp16_row_np.astype(np.float32).tolist()
            fp16_bits = fp16_row_np.view(np.uint16)
            if self._fp16_to_fx_lut_np is not None:
                fx_raw_np = self._fp16_to_fx_lut_np[fp16_bits]
            else:
                fx_raw_np = np.asarray(
                    [rtl_m1_fp16_to_base2_q6_10(int(bits)) for bits in fp16_bits.tolist()],
                    dtype=np.int32,
                )
            fx_raw = fx_raw_np.astype(np.int32).tolist()
            fx_values = (fx_raw_np.astype(np.float32) / float(1 << self.cfg.fx_frac_width)).tolist()
            return fp_quantized, fx_raw, fx_values

        fp_quantized = []
        fx_raw = []
        fx_values = []
        for value in row:
            fp_value = quantize_input_format(value, self.cfg.input_format)
            if math.isnan(fp_value):
                fp_value = 0.0
            fp_quantized.append(fp_value)
            if self.rtl_exact_enabled():
                base2_raw = rtl_m1_fp16_to_base2_q6_10(float_to_fp16_bits(fp_value))
            else:
                input_raw = self.cfg.fx_fmt.quantize(fp_value)
                base2_raw = clamp_raw(shift_add_multiply(input_raw, self.cfg.log2e_shift_terms), self.cfg.fx_fmt)
            fx_raw.append(base2_raw)
            fx_values.append(self.cfg.fx_fmt.to_float(base2_raw))
        return fp_quantized, fx_raw, fx_values

    def module2_block_load_max(self, fx_raw_row: Sequence[int]) -> Tuple[List[int], List[int]]:
        block_max_integer_parts: List[int] = []
        block_max_values: List[int] = []
        for block in split_blocks(fx_raw_row, self.cfg.block_size_effective):
            block_max_raw = max(block)
            block_max_integer_parts.append(fixed_floor_int(block_max_raw, self.cfg.fx_frac_width))
            block_max_values.append(fixed_ceil_int(block_max_raw, self.cfg.fx_frac_width))
        return block_max_integer_parts, block_max_values

    def module3_block_prune_y_generate(
        self, fx_raw_row: Sequence[int], block_max_values: Sequence[int]
    ) -> Tuple[List[BlockMeta], List[SoftmaxDescriptor], List[int], List[float], List[bool], List[bool], List[float]]:
        y_raw_row: List[int] = []
        y_row: List[float] = []
        descriptors: List[SoftmaxDescriptor] = []
        metas: List[BlockMeta] = []
        block_prune_flags: List[bool] = []
        block_prune_deltas: List[float] = []
        element_prune_flags: List[bool] = []
        prune_flags: List[bool] = []
        prune_compare_mode = self.cfg.elem_prune_compare_mode
        if prune_compare_mode not in ("full_y", "floor_int", "trunc_int"):
            raise ValueError(f"unsupported elem_prune_compare_mode: {prune_compare_mode}")
        running_max: int | None = None
        elem_cursor = 0

        for block_index, block in enumerate(split_blocks(fx_raw_row, self.cfg.block_size_effective)):
            block_max_value = block_max_values[block_index]
            tau_elem, tau_elem_int, block_threshold, c2_count = self.resolve_block_prune_policy(
                block, block_max_value, prune_compare_mode
            )
            if block_index == 0 or not self.cfg.block_prune_enabled or running_max is None:
                block_prune_delta = 0.0
                block_pruned = False
            else:
                block_prune_delta = float(block_max_value - running_max)
                block_pruned = block_prune_delta <= block_threshold
            block_prune_deltas.append(block_prune_delta)
            block_prune_flags.append(block_pruned)

            descriptor_start_index = elem_cursor
            element_count = len(block)
            element_threshold_int = block_max_value + tau_elem_int
            prune_raw = self.cfg.y_fmt.quantize(tau_elem)
            for fx_raw in block:
                y_base2_raw = fx_raw - (block_max_value << self.cfg.fx_frac_width)
                if self.rtl_exact_enabled():
                    y_raw = rescale_signed_raw(y_base2_raw, self.cfg.fx_frac_width, self.cfg.y_frac_width)
                else:
                    y_value = self.cfg.fx_fmt.to_float(y_base2_raw)
                    y_raw = self.cfg.y_fmt.quantize(y_value)
                y_raw_row.append(y_raw)
                y_row.append(self.cfg.y_fmt.to_float(y_raw))

                if prune_compare_mode == "full_y":
                    element_pruned = y_raw < prune_raw
                else:
                    element_pruned = self.element_prune_integer_extract(fx_raw, prune_compare_mode) < element_threshold_int
                element_prune_flags.append(element_pruned)
                final_pruned = block_pruned or element_pruned
                prune_flags.append(final_pruned)

                if final_pruned:
                    descriptors.append(SoftmaxDescriptor(y_shift=0, y_frac_raw=0, y_prune=True))
                else:
                    descriptors.append(self._build_live_descriptor(y_raw))
                elem_cursor += 1

            metas.append(
                BlockMeta(
                    row_id=0,
                    block_index=block_index,
                    block_max_value=block_max_value,
                    block_pruned=block_pruned,
                    block_sum_raw=0,
                    descriptor_start_index=descriptor_start_index,
                    element_count=element_count,
                    partial_valid=True,
                    complete=False,
                    tau_elem_value=tau_elem,
                    tau_blk_value=block_threshold,
                    c2_count=c2_count,
                )
            )
            running_max = block_max_value if running_max is None else max(running_max, block_max_value)

        return metas, descriptors, y_raw_row, y_row, element_prune_flags, prune_flags, block_prune_deltas

    def module4_exp_block_reduce(
        self,
        descriptors: Sequence[SoftmaxDescriptor],
        metas: Sequence[BlockMeta],
        y_raw_row: Sequence[int],
    ) -> Tuple[List[BlockMeta], List[int]]:
        completed_metas: List[BlockMeta] = []
        block_sum_raw_values: List[int] = []
        for meta in metas:
            if meta.block_pruned:
                block_sum_raw = 0
            else:
                block_sum_raw = 0
                start = meta.descriptor_start_index
                stop = meta.descriptor_start_index + meta.element_count
                for elem_index in range(start, stop):
                    desc = descriptors[elem_index]
                    if desc.y_prune:
                        continue
                    if self.cfg.sum_use_buffered_descriptor:
                        y_shift = desc.y_shift
                        frac_raw = desc.y_frac_raw
                        frac_width = self.cfg.y_frac_buf_width
                    else:
                        y_raw = y_raw_row[elem_index]
                        y_floor = fixed_floor_int(y_raw, self.cfg.y_frac_width)
                        frac_raw = y_raw - (y_floor << self.cfg.y_frac_width)
                        y_shift = max(0, min((1 << self.cfg.y_shift_width) - 1, -y_floor))
                        frac_width = self.cfg.y_frac_width
                    mantissa_src_raw = approx_exp2_frac_raw(frac_raw, frac_width, self.cfg.exp_pwl_mode)
                    mantissa_raw = rescale_frac_raw(mantissa_src_raw, frac_width, self.cfg.exp_frac_width)
                    exp_raw = mantissa_raw >> y_shift
                    exp_sum_raw = rescale_frac_raw(exp_raw, self.cfg.exp_frac_width, self.cfg.sum_frac_width)
                    block_sum_raw = min(self.cfg.sum_fmt.max_raw, block_sum_raw + exp_sum_raw)
            block_sum_raw_values.append(block_sum_raw)
            completed_metas.append(
                BlockMeta(
                    row_id=meta.row_id,
                    block_index=meta.block_index,
                    block_max_value=meta.block_max_value,
                    block_pruned=meta.block_pruned,
                    block_sum_raw=block_sum_raw,
                    descriptor_start_index=meta.descriptor_start_index,
                    element_count=meta.element_count,
                    partial_valid=True,
                    complete=True,
                    tau_elem_value=meta.tau_elem_value,
                    tau_blk_value=meta.tau_blk_value,
                    c2_count=meta.c2_count,
                )
            )
        return completed_metas, block_sum_raw_values

    def module5_row_online_merge(self, metas: Sequence[BlockMeta]) -> RowFinalStateToken:
        if not metas:
            return RowFinalStateToken(row_id=0, row_bank=0, row_max_value_final=0, row_sum_final_raw=0, block_count=0)
        max_accumulator = metas[0].block_max_value
        sum_accumulator = metas[0].block_sum_raw
        for meta in metas[1:]:
            if meta.block_pruned:
                continue
            if meta.block_max_value >= max_accumulator:
                sum_accumulator = (sum_accumulator >> (meta.block_max_value - max_accumulator)) + meta.block_sum_raw
                max_accumulator = meta.block_max_value
            else:
                sum_accumulator = sum_accumulator + (meta.block_sum_raw >> (max_accumulator - meta.block_max_value))
        return RowFinalStateToken(
            row_id=metas[0].row_id,
            row_bank=0,
            row_max_value_final=max_accumulator,
            row_sum_final_raw=sum_accumulator,
            block_count=len(metas),
        )

    def module6_row_denom_prep(
        self, row_final_state_token: RowFinalStateToken
    ) -> Tuple[int, float, int, int, float, int, float, RowNormalizationToken]:
        if row_final_state_token.row_sum_final_raw <= 0:
            zero_token = RowNormalizationToken(
                row_id=row_final_state_token.row_id,
                row_bank=row_final_state_token.row_bank,
                row_max_value_final=row_final_state_token.row_max_value_final,
                normalized_row_sum_exponent=0,
                denom_k_raw=0,
                denom_k=0.0,
                denom_delta_raw=0,
                denom_delta=0.0,
                block_count=row_final_state_token.block_count,
            )
            return 0, 0.0, 0, 0, 0.0, 0, 0.0, zero_token

        if self.rtl_exact_enabled() and row_final_state_token.row_sum_final_raw < (1 << self.cfg.sum_frac_width):
            normalized_row_sum_mantissa_raw = row_final_state_token.row_sum_final_raw << 1
            normalized_row_sum_exponent = -1
        else:
            normalized_row_sum_mantissa_raw, normalized_row_sum_exponent = normalize_sum(
                row_final_state_token.row_sum_final_raw, self.cfg.sum_frac_width
            )
        normalized_row_sum_mantissa = normalized_row_sum_mantissa_raw / (1 << self.cfg.sum_frac_width)
        if self.rtl_exact_enabled():
            denom_k_raw = normalized_row_sum_mantissa_raw & ((1 << self.cfg.sum_frac_width) - 1)
        else:
            unit_raw = 1 << self.cfg.sum_frac_width
            denom_k_raw = max(0, normalized_row_sum_mantissa_raw - unit_raw)
        denom_k = denom_k_raw / (1 << self.cfg.sum_frac_width)
        denom_delta_raw = approx_delta_triangle_raw(denom_k_raw, self.cfg.sum_frac_width)
        denom_delta = denom_delta_raw / (1 << self.cfg.sum_frac_width)

        normalization_token = RowNormalizationToken(
            row_id=row_final_state_token.row_id,
            row_bank=row_final_state_token.row_bank,
            row_max_value_final=row_final_state_token.row_max_value_final,
            normalized_row_sum_exponent=normalized_row_sum_exponent,
            denom_k_raw=denom_k_raw,
            denom_k=denom_k,
            denom_delta_raw=denom_delta_raw,
            denom_delta=denom_delta,
            block_count=row_final_state_token.block_count,
        )
        return (
            normalized_row_sum_mantissa_raw,
            normalized_row_sum_mantissa,
            normalized_row_sum_exponent,
            denom_k_raw,
            denom_k,
            denom_delta_raw,
            denom_delta,
            normalization_token,
        )

    def module6_replay_output(
        self,
        descriptors: Sequence[SoftmaxDescriptor],
        metas: Sequence[BlockMeta],
        row_normalization_token: RowNormalizationToken,
    ) -> Tuple[List[float], List[int]]:
        approx_probs: List[float] = []
        approx_probs_raw: List[int] = []
        out_scale = 1 << self.cfg.out_frac_width
        out_shift = self.cfg.out_frac_width - self.cfg.exp_frac_width
        for meta in metas:
            block_delta = row_normalization_token.row_max_value_final - meta.block_max_value
            for desc in descriptors[meta.descriptor_start_index : meta.descriptor_start_index + meta.element_count]:
                if desc.y_prune:
                    approx_probs_raw.append(0)
                    approx_probs.append(0.0)
                    continue
                if self.rtl_exact_enabled():
                    merged_exp_q5 = desc.y_frac_raw - row_normalization_token.denom_k_raw - row_normalization_token.denom_delta_raw
                    if merged_exp_q5 < 0:
                        lut_in_q5 = merged_exp_q5 + (1 << self.cfg.sum_frac_width)
                        exp_rshift_extra = 1
                    else:
                        lut_in_q5 = merged_exp_q5 & ((1 << self.cfg.sum_frac_width) - 1)
                        exp_rshift_extra = 0
                    exp_term = approx_exp2_frac_raw(lut_in_q5, self.cfg.sum_frac_width, self.cfg.exp_pwl_mode)
                    final_rshift = (
                        desc.y_shift
                        + block_delta
                        + row_normalization_token.normalized_row_sum_exponent
                        + exp_rshift_extra
                    )
                    if final_rshift < 0:
                        prob_raw = (exp_term << (-final_rshift)) << out_shift
                    else:
                        prob_raw = (exp_term >> final_rshift) << out_shift
                    prob_raw &= (1 << self.cfg.out_fmt.total_width) - 1
                else:
                    v_frac_raw = rescale_frac_raw(desc.y_frac_raw, self.cfg.y_frac_buf_width, self.cfg.sum_frac_width)
                    v_internal_raw = v_frac_raw - row_normalization_token.denom_k_raw - row_normalization_token.denom_delta_raw
                    v_internal_floor = fixed_floor_int(v_internal_raw, self.cfg.sum_frac_width)
                    v_internal_frac_raw = v_internal_raw - (v_internal_floor << self.cfg.sum_frac_width)
                    mantissa_raw = approx_exp2_frac_raw(
                        v_internal_frac_raw,
                        self.cfg.sum_frac_width,
                        self.cfg.exp_pwl_mode,
                    )
                    total_shift = (
                        desc.y_shift
                        + block_delta
                        + row_normalization_token.normalized_row_sum_exponent
                        + (-v_internal_floor)
                    )
                    prob_value = (mantissa_raw / (1 << self.cfg.sum_frac_width)) / (2 ** total_shift)
                    prob_raw = quantize_output_probability(prob_value, self.cfg.out_frac_width)
                approx_probs_raw.append(prob_raw)
                approx_probs.append(prob_raw / out_scale)
        return approx_probs, approx_probs_raw

    def simulate_row(self, row: Sequence[float]) -> SoftmaxRowResult:
        fp_quantized_row, fx_raw_row, fx_row = self.module1_fp_to_fx(row)
        block_max_integer_parts, block_max_values = self.module2_block_load_max(fx_raw_row)
        (
            partial_metas,
            descriptors,
            y_raw_row,
            y_row,
            element_prune_flags,
            prune_flags,
            block_prune_deltas,
        ) = self.module3_block_prune_y_generate(fx_raw_row, block_max_values)
        block_prune_flags = [meta.block_pruned for meta in partial_metas]
        completed_metas, block_sum_raw_values = self.module4_exp_block_reduce(descriptors, partial_metas, y_raw_row)
        row_final_state_token = self.module5_row_online_merge(completed_metas)
        (
            normalized_row_sum_mantissa_raw,
            normalized_row_sum_mantissa,
            normalized_row_sum_exponent,
            denom_k_raw,
            denom_k,
            denom_delta_raw,
            denom_delta,
            row_normalization_token,
        ) = self.module6_row_denom_prep(row_final_state_token)
        approx_probs, approx_probs_raw = self.module6_replay_output(descriptors, completed_metas, row_normalization_token)
        reference_probs = exact_softmax(fp_quantized_row)
        abs_errors = [abs(approx - ref) for approx, ref in zip(approx_probs, reference_probs)]
        sq_errors = [(approx - ref) ** 2 for approx, ref in zip(approx_probs, reference_probs)]
        return SoftmaxRowResult(
            input_row=list(row),
            fp_quantized_row=fp_quantized_row,
            fx_raw_row=fx_raw_row,
            fx_row=fx_row,
            block_max_integer_parts=block_max_integer_parts,
            block_max_values=block_max_values,
            block_prune_deltas=block_prune_deltas,
            row_max_value_final=row_final_state_token.row_max_value_final,
            block_prune_flags=list(block_prune_flags),
            y_raw_row=y_raw_row,
            y_row=y_row,
            element_prune_flags=list(element_prune_flags),
            prune_flags=list(prune_flags),
            descriptors=descriptors,
            block_metas=completed_metas,
            block_sum_raw_values=block_sum_raw_values,
            row_sum_final_raw=row_final_state_token.row_sum_final_raw,
            row_sum_final=self.cfg.sum_fmt.to_float(row_final_state_token.row_sum_final_raw),
            row_final_state_token=row_final_state_token,
            row_normalization_token=row_normalization_token,
            normalized_row_sum_mantissa_raw=normalized_row_sum_mantissa_raw,
            normalized_row_sum_mantissa=normalized_row_sum_mantissa,
            normalized_row_sum_exponent=normalized_row_sum_exponent,
            denom_k_raw=denom_k_raw,
            denom_k=denom_k,
            denom_delta_raw=denom_delta_raw,
            denom_delta=denom_delta,
            approx_probs=approx_probs,
            approx_probs_raw=approx_probs_raw,
            reference_probs=reference_probs,
            max_abs_error=max(abs_errors) if abs_errors else 0.0,
            mean_abs_error=sum(abs_errors) / len(abs_errors) if abs_errors else 0.0,
            mse=sum(sq_errors) / len(sq_errors) if sq_errors else 0.0,
            kl_divergence=calculate_kl_divergence(reference_probs, approx_probs),
        )

    def _supports_batch_fast_path(self, row_depth: int) -> bool:
        return (
            self._fast_path_frac_width() is not None
            and row_depth > 0
            and (row_depth % self.cfg.block_size_effective) == 0
        )

    def _supports_torch_batch_fast_path(self, row_depth: int) -> bool:
        return (
            self._fast_path_frac_width() == 5
            and self.cfg.descriptor_mode == "shift_frac"
            and row_depth > 0
            and (row_depth % self.cfg.block_size_effective) == 0
        )

    def _fast_path_frac_width(self) -> int | None:
        base_supported = (
            np is not None
            and self.rtl_exact_enabled()
            and self.cfg.input_format == "fp16"
            and self.cfg.sum_use_buffered_descriptor
            and self.cfg.exp_pwl_mode == "DOC"
            and self.cfg.elem_prune_compare_mode in ("floor_int", "trunc_int")
            and self.cfg.adaptive_prune_mode in ("fixed", "block_c2_two_level")
            and self.cfg.block_size_effective > 0
            and self.cfg.y_frac_buf_width == self.cfg.exp_frac_width == self.cfg.sum_frac_width
        )
        if not base_supported:
            return None
        special_frac_width = special_descriptor_frac_width(self.cfg.descriptor_mode)
        if (
            special_frac_width is not None
            and self.cfg.y_frac_width == special_frac_width
            and self.cfg.y_frac_buf_width == special_frac_width
            and self.cfg.y_shift_width == 2
        ):
            return special_frac_width
        return None

    def _build_buffered_descriptor_fast_np(
        self,
        y_raw: "np.ndarray",
        final_pruned: "np.ndarray",
    ) -> Tuple["np.ndarray", "np.ndarray"]:
        desc_shift = np.zeros_like(y_raw, dtype=np.int32)
        desc_frac = np.zeros_like(y_raw, dtype=np.int32)
        live_mask = ~final_pruned
        if not np.any(live_mask):
            return desc_shift, desc_frac

        unit_raw = 1 << self.cfg.y_frac_width
        if special_descriptor_frac_width(self.cfg.descriptor_mode) is not None:
            z_mag_raw = np.maximum(0, -y_raw).astype(np.int32)
            special_mask = live_mask & (z_mag_raw >= (4 * unit_raw) - 1)
            regular_mask = live_mask & ~special_mask & (z_mag_raw != 0)
            if np.any(regular_mask):
                regular_z = z_mag_raw[regular_mask]
                regular_shift = (regular_z + unit_raw - 1) >> self.cfg.y_frac_width
                desc_shift[regular_mask] = regular_shift
                desc_frac[regular_mask] = (regular_shift << self.cfg.y_frac_width) - regular_z
            if np.any(special_mask):
                desc_shift[special_mask] = 4
                desc_frac[special_mask] = 1
            return desc_shift, desc_frac

        shift_abs = np.zeros_like(y_raw, dtype=np.int32)
        shift_abs[live_mask] = ((-y_raw[live_mask]) + unit_raw - 1) >> self.cfg.y_frac_width
        frac_raw = np.zeros_like(y_raw, dtype=np.int32)
        frac_raw[live_mask] = y_raw[live_mask] + (shift_abs[live_mask] << self.cfg.y_frac_width)
        desc_shift = shift_abs & ((1 << self.cfg.y_shift_width) - 1)
        desc_frac = frac_raw & ((1 << self.cfg.y_frac_buf_width) - 1)
        return desc_shift.astype(np.int32), desc_frac.astype(np.int32)

    def _approx_exp_q5_array(self, frac_q5: "np.ndarray") -> "np.ndarray":
        tri_term = np.where(frac_q5 < 16, frac_q5, 32 - frac_q5)
        corr_term = (tri_term >> 3) + (tri_term >> 4)
        return 32 + frac_q5 - corr_term

    def _approx_exp_q5_tensor(self, frac_q5: "torch.Tensor") -> "torch.Tensor":
        tri_term = torch.where(frac_q5 < 16, frac_q5, 32 - frac_q5)
        corr_term = (tri_term >> 3) + (tri_term >> 4)
        return 32 + frac_q5 - corr_term

    def _zero_prune_stats(self) -> Dict[str, int]:
        return {
            "total_elements": 0,
            "total_pruned_elements": 0,
            "block_pruned_blocks": 0,
            "total_blocks": 0,
            "block_pruned_elements": 0,
            "element_pruned_only": 0,
            "aggressive_blocks": 0,
            "conservative_blocks": 0,
        }

    def _fast_path_compare_values_np(self, fx_blocks: "np.ndarray") -> "np.ndarray":
        if self.cfg.elem_prune_compare_mode == "floor_int":
            return fx_blocks >> self.cfg.fx_frac_width
        if self.cfg.elem_prune_compare_mode == "trunc_int":
            return np.where(
                fx_blocks >= 0,
                fx_blocks >> self.cfg.fx_frac_width,
                -((-fx_blocks) >> self.cfg.fx_frac_width),
            )
        raise ValueError(f"unsupported fast-path compare mode: {self.cfg.elem_prune_compare_mode}")

    def _fast_path_compare_values_torch(self, fx_blocks: "torch.Tensor", int_dtype) -> "torch.Tensor":
        if self.cfg.elem_prune_compare_mode == "floor_int":
            return fx_blocks >> self.cfg.fx_frac_width
        if self.cfg.elem_prune_compare_mode == "trunc_int":
            return torch.where(
                fx_blocks >= 0,
                fx_blocks >> self.cfg.fx_frac_width,
                -((-fx_blocks) >> self.cfg.fx_frac_width),
            ).to(int_dtype)
        raise ValueError(f"unsupported fast-path compare mode: {self.cfg.elem_prune_compare_mode}")

    def _resolve_fast_path_tau_np(
        self,
        compare_values: "np.ndarray",
        block_max_int: "np.ndarray",
        block_clog2: int,
    ) -> Tuple["np.ndarray", "np.ndarray", "np.ndarray | None"]:
        if self.cfg.adaptive_prune_mode == "fixed":
            tau_elem = np.full_like(block_max_int, self.cfg.prune_threshold_int, dtype=np.int32)
            use_aggressive = None
        else:
            c2_count = (compare_values >= (block_max_int[:, :, None] - 2)).sum(axis=2)
            use_aggressive = c2_count <= self.cfg.adaptive_c2_threshold
            tau_elem = np.where(
                use_aggressive,
                self.cfg.adaptive_tau_aggressive_int,
                self.cfg.adaptive_tau_conservative_int,
            ).astype(np.int32)
        if self.cfg.block_prune_threshold is not None:
            tau_blk = np.full_like(block_max_int, int(round(self.cfg.block_prune_threshold)), dtype=np.int32)
        else:
            tau_blk = np.full_like(
                block_max_int,
                int(round(self.cfg.block_prune_threshold_effective)),
                dtype=np.int32,
            )
        return tau_elem, tau_blk.astype(np.int32), use_aggressive

    def _resolve_fast_path_tau_torch(
        self,
        compare_values: "torch.Tensor",
        block_max_int: "torch.Tensor",
        block_clog2: int,
        int_dtype,
    ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor | None"]:
        if self.cfg.adaptive_prune_mode == "fixed":
            tau_elem = torch.full_like(block_max_int, self.cfg.prune_threshold_int, dtype=int_dtype)
            use_aggressive = None
        else:
            c2_count = (compare_values >= (block_max_int.unsqueeze(2) - 2)).sum(dim=2).to(int_dtype)
            use_aggressive = c2_count <= self.cfg.adaptive_c2_threshold
            tau_elem = torch.where(
                use_aggressive,
                torch.full_like(block_max_int, self.cfg.adaptive_tau_aggressive_int, dtype=int_dtype),
                torch.full_like(block_max_int, self.cfg.adaptive_tau_conservative_int, dtype=int_dtype),
            )
        if self.cfg.block_prune_threshold is not None:
            tau_blk = torch.full_like(block_max_int, int(round(self.cfg.block_prune_threshold)), dtype=int_dtype)
        else:
            tau_blk = torch.full_like(
                block_max_int,
                int(round(self.cfg.block_prune_threshold_effective)),
                dtype=int_dtype,
            )
        return tau_elem, tau_blk.to(int_dtype), use_aggressive

    def _aggregate_row_fast_stats(self, rows: Sequence[Sequence[float]] | "np.ndarray" | "torch.Tensor") -> Tuple[List[List[float]], Dict[str, int]]:
        approx_rows: List[List[float]] = []
        agg_stats = self._zero_prune_stats()
        for row in rows:
            if torch is not None and isinstance(row, torch.Tensor):
                row_values = row.tolist()
            elif np is not None and isinstance(row, np.ndarray):
                row_values = row.tolist()
            else:
                row_values = list(row)
            _, approx_probs, prune_stats = self.simulate_row_fast(row_values)
            approx_rows.append(approx_probs)
            for key in agg_stats:
                agg_stats[key] += prune_stats[key]
        return approx_rows, agg_stats

    def _simulate_rows_fast_torch(self, rows_tensor: "torch.Tensor") -> Tuple["torch.Tensor", Dict[str, int]]:
        rows_tensor = rows_tensor.to(dtype=torch.float32, device="cpu")
        if rows_tensor.ndim != 2 or rows_tensor.shape[0] == 0:
            return torch.zeros((0, 0), dtype=torch.float32), self._zero_prune_stats()

        row_count, row_depth = rows_tensor.shape
        block_size = self.cfg.block_size_effective
        block_count = row_depth // block_size
        block_clog2 = self.cfg.block_clog2
        out_shift = self.cfg.out_frac_width - self.cfg.exp_frac_width
        out_scale = float(1 << self.cfg.out_frac_width)
        int_dtype = torch.int32

        with torch.no_grad():
            fp16_rows = rows_tensor.to(torch.float16)
            fp16_bits = fp16_rows.view(torch.int16).to(torch.int32) & 0xFFFF
            sign_bits = fp16_bits >> 15
            exp_bits = (fp16_bits >> 10) & 0x1F
            frac_bits = fp16_bits & 0x3FF

            q6_10_raw = torch.zeros_like(exp_bits, dtype=int_dtype)
            inf_mask = exp_bits == 0x1F
            q6_10_raw[inf_mask] = torch.where(
                sign_bits[inf_mask] != 0,
                torch.full_like(sign_bits[inf_mask], -32768, dtype=int_dtype),
                torch.full_like(sign_bits[inf_mask], 32767, dtype=int_dtype),
            )

            normal_mask = (exp_bits != 0) & (exp_bits != 0x1F)
            if torch.any(normal_mask):
                unbiased_exp = exp_bits[normal_mask] - 15
                mantissa = 1024 + frac_bits[normal_mask]
                normal_raw = torch.empty_like(mantissa, dtype=int_dtype)
                left_mask = unbiased_exp >= 0
                normal_raw[left_mask] = mantissa[left_mask] << unbiased_exp[left_mask]
                normal_raw[~left_mask] = mantissa[~left_mask] >> (-unbiased_exp[~left_mask])
                signed_mask = sign_bits[normal_mask] != 0
                normal_raw[signed_mask] = -normal_raw[signed_mask]
                q6_10_raw[normal_mask] = normal_raw

            q6_10_raw.clamp_(-32768, 32767)
            fx_raw = q6_10_raw + (q6_10_raw >> 1) - (q6_10_raw >> 4)
            fx_raw.clamp_(-32768, 32767)

            fx_blocks = fx_raw.reshape(row_count, block_count, block_size)
            block_max_raw = fx_blocks.max(dim=2).values
            m_local = (block_max_raw >> self.cfg.fx_frac_width) + (
                (block_max_raw & ((1 << self.cfg.fx_frac_width) - 1)) != 0
            ).to(int_dtype)

            compare_values = self._fast_path_compare_values_torch(fx_blocks, int_dtype)
            tau_elem, tau_blk, use_aggressive = self._resolve_fast_path_tau_torch(
                compare_values,
                m_local,
                block_clog2,
                int_dtype,
            )

            prev_running_max = torch.zeros_like(m_local, dtype=int_dtype)
            if block_count > 1:
                prev_running_max[:, 1:] = torch.cummax(m_local[:, :-1], dim=1).values

            block_pruned = torch.zeros_like(m_local, dtype=torch.bool)
            if self.cfg.block_prune_enabled and block_count > 1:
                block_pruned[:, 1:] = (m_local[:, 1:] - prev_running_max[:, 1:]) <= tau_blk[:, 1:]

            element_threshold = m_local.unsqueeze(2) + tau_elem.unsqueeze(2)
            element_pruned = compare_values < element_threshold
            final_pruned = block_pruned.unsqueeze(2) | element_pruned

            y_raw = (fx_blocks - (m_local.unsqueeze(2) << self.cfg.fx_frac_width)) >> (
                self.cfg.fx_frac_width - self.cfg.y_frac_width
            )
            shift_abs = torch.where(
                final_pruned,
                torch.zeros_like(y_raw),
                ((-y_raw) + ((1 << self.cfg.y_frac_width) - 1)) >> self.cfg.y_frac_width,
            )
            frac_raw = torch.where(
                final_pruned,
                torch.zeros_like(y_raw),
                y_raw + (shift_abs << self.cfg.y_frac_width),
            )
            desc_shift = shift_abs & ((1 << self.cfg.y_shift_width) - 1)
            desc_frac = frac_raw & ((1 << self.cfg.y_frac_buf_width) - 1)

            exp_local = torch.where(
                final_pruned,
                torch.zeros_like(desc_frac),
                self._approx_exp_q5_tensor(desc_frac) >> desc_shift,
            ).to(int_dtype)
            sum_local = exp_local.sum(dim=2).to(int_dtype)

            row_max_final = m_local[:, 0].clone()
            row_sum_final = sum_local[:, 0].clone()
            for block_idx in range(1, block_count):
                live_mask = ~block_pruned[:, block_idx]
                if not torch.any(live_mask):
                    continue
                use_new_max = live_mask & (m_local[:, block_idx] >= row_max_final)
                if torch.any(use_new_max):
                    delta = m_local[use_new_max, block_idx] - row_max_final[use_new_max]
                    row_sum_final[use_new_max] = (row_sum_final[use_new_max] >> delta) + sum_local[use_new_max, block_idx]
                    row_max_final[use_new_max] = m_local[use_new_max, block_idx]
                keep_old_max = live_mask & ~use_new_max
                if torch.any(keep_old_max):
                    delta = row_max_final[keep_old_max] - m_local[keep_old_max, block_idx]
                    row_sum_final[keep_old_max] = row_sum_final[keep_old_max] + (sum_local[keep_old_max, block_idx] >> delta)

            denom_exp = torch.zeros(row_count, dtype=int_dtype)
            denom_k_raw = torch.zeros(row_count, dtype=int_dtype)
            positive_sum_mask = row_sum_final > 0
            if torch.any(positive_sum_mask):
                small_sum_mask = positive_sum_mask & (row_sum_final < (1 << self.cfg.sum_frac_width))
                denom_exp[small_sum_mask] = -1
                denom_k_raw[small_sum_mask] = (row_sum_final[small_sum_mask] << 1) & ((1 << self.cfg.sum_frac_width) - 1)

                normal_sum_mask = positive_sum_mask & ~small_sum_mask
                if torch.any(normal_sum_mask):
                    highest_bit = torch.floor(torch.log2(row_sum_final[normal_sum_mask].to(torch.float32))).to(int_dtype)
                    denom_exp[normal_sum_mask] = highest_bit - self.cfg.sum_frac_width
                    norm_raw = row_sum_final[normal_sum_mask] >> denom_exp[normal_sum_mask]
                    denom_k_raw[normal_sum_mask] = norm_raw & ((1 << self.cfg.sum_frac_width) - 1)

            denom_fold = torch.where(denom_k_raw < 16, denom_k_raw, 32 - denom_k_raw)
            denom_delta_raw = ((denom_fold * 3) + 8) >> 4
            block_delta = row_max_final.unsqueeze(1) - m_local

            merged_exp = desc_frac.to(int_dtype) - denom_k_raw[:, None, None] - denom_delta_raw[:, None, None]
            exp_rshift_extra = merged_exp < 0
            lut_in_q5 = torch.where(
                exp_rshift_extra,
                merged_exp + (1 << self.cfg.sum_frac_width),
                merged_exp & ((1 << self.cfg.sum_frac_width) - 1),
            )
            exp_term = self._approx_exp_q5_tensor(lut_in_q5).to(int_dtype)
            final_rshift = (
                desc_shift.to(int_dtype)
                + block_delta.unsqueeze(2)
                + denom_exp[:, None, None]
                + exp_rshift_extra.to(int_dtype)
            )

            shift_left = (-final_rshift).clamp_min(0)
            shift_right = final_rshift.clamp_min(0)
            shifted_exp = torch.where(
                final_rshift < 0,
                exp_term << shift_left,
                exp_term >> shift_right,
            )
            out_raw = torch.where(final_pruned, torch.zeros_like(shifted_exp), shifted_exp << out_shift)
            out_raw &= (1 << self.cfg.out_fmt.total_width) - 1

            approx_probs = out_raw.reshape(row_count, row_depth).to(torch.float32) / out_scale
            prune_stats = {
                "total_elements": int(row_count * row_depth),
                "total_pruned_elements": int(final_pruned.sum().item()),
                "block_pruned_blocks": int(block_pruned.sum().item()),
                "total_blocks": int(row_count * block_count),
                "block_pruned_elements": int(block_pruned.sum().item() * block_size),
                "element_pruned_only": int((element_pruned & ~block_pruned.unsqueeze(2)).sum().item()),
                "aggressive_blocks": 0 if use_aggressive is None else int(use_aggressive.sum().item()),
                "conservative_blocks": (
                    0 if use_aggressive is None else int((~use_aggressive).sum().item())
                ),
            }
            return approx_probs, prune_stats

    def simulate_rows_fast(self, rows: Sequence[Sequence[float]] | "np.ndarray") -> Tuple["np.ndarray", Dict[str, int]]:
        if torch is not None and isinstance(rows, torch.Tensor):
            if rows.device.type == "cpu":
                return self.simulate_rows_fast(rows.detach().to(dtype=torch.float32).numpy())  # type: ignore[return-value]
            row_depth = int(rows.shape[-1]) if rows.ndim == 2 else 0
            if self._supports_torch_batch_fast_path(row_depth):
                return self._simulate_rows_fast_torch(rows)  # type: ignore[return-value]
            return self._aggregate_row_fast_stats(rows)  # type: ignore[return-value]

        if np is None:
            return self._aggregate_row_fast_stats(rows)  # type: ignore[return-value]

        rows_np = np.asarray(rows, dtype=np.float32)
        if rows_np.ndim != 2 or rows_np.shape[0] == 0:
            return np.zeros((0, 0), dtype=np.float32), self._zero_prune_stats()

        row_count, row_depth = rows_np.shape
        if not self._supports_batch_fast_path(row_depth):
            approx_rows, agg_stats = self._aggregate_row_fast_stats(rows_np)
            return np.asarray(approx_rows, dtype=np.float32), agg_stats

        frac_width = self._fast_path_frac_width()
        if frac_width is None:
            approx_rows, agg_stats = self._aggregate_row_fast_stats(rows_np)
            return np.asarray(approx_rows, dtype=np.float32), agg_stats

        block_size = self.cfg.block_size_effective
        block_count = row_depth // block_size
        block_clog2 = self.cfg.block_clog2
        out_shift = self.cfg.out_frac_width - self.cfg.exp_frac_width
        out_scale = float(1 << self.cfg.out_frac_width)

        with np.errstate(over="ignore"):
            fp16_bits = rows_np.astype(np.float16).view(np.uint16)
        if self._fp16_to_fx_lut_np is not None:
            fx_raw = self._fp16_to_fx_lut_np[fp16_bits]
        else:
            sign_bits = (fp16_bits >> 15).astype(np.int32)
            exp_bits = ((fp16_bits >> 10) & 0x1F).astype(np.int32)
            frac_bits = (fp16_bits & 0x3FF).astype(np.int32)

            q6_10_raw = np.zeros_like(exp_bits, dtype=np.int32)
            inf_mask = exp_bits == 0x1F
            q6_10_raw[inf_mask] = np.where(sign_bits[inf_mask] != 0, -32768, 32767)

            normal_mask = (exp_bits != 0) & (exp_bits != 0x1F)
            if np.any(normal_mask):
                unbiased_exp = exp_bits[normal_mask] - 15
                mantissa = 1024 + frac_bits[normal_mask]
                normal_raw = np.empty_like(mantissa, dtype=np.int32)
                left_mask = unbiased_exp >= 0
                normal_raw[left_mask] = mantissa[left_mask] << unbiased_exp[left_mask]
                normal_raw[~left_mask] = mantissa[~left_mask] >> (-unbiased_exp[~left_mask])
                signed_mask = sign_bits[normal_mask] != 0
                normal_raw[signed_mask] = -normal_raw[signed_mask]
                q6_10_raw[normal_mask] = normal_raw

            np.clip(q6_10_raw, -32768, 32767, out=q6_10_raw)
            fx_raw = q6_10_raw + (q6_10_raw >> 1) - (q6_10_raw >> 4)
            np.clip(fx_raw, -32768, 32767, out=fx_raw)

        fx_blocks = fx_raw.reshape(row_count, block_count, block_size)
        block_max_raw = fx_blocks.max(axis=2)
        m_local = (block_max_raw >> self.cfg.fx_frac_width) + (
            (block_max_raw & ((1 << self.cfg.fx_frac_width) - 1)) != 0
        ).astype(np.int32)

        compare_values = self._fast_path_compare_values_np(fx_blocks)
        tau_elem, tau_blk, use_aggressive = self._resolve_fast_path_tau_np(
            compare_values,
            m_local,
            block_clog2,
        )

        prev_running_max = np.zeros_like(m_local, dtype=np.int32)
        if block_count > 1:
            prev_running_max[:, 1:] = np.maximum.accumulate(m_local[:, :-1], axis=1)
        block_pruned = np.zeros_like(m_local, dtype=bool)
        if self.cfg.block_prune_enabled and block_count > 1:
            block_pruned[:, 1:] = (m_local[:, 1:] - prev_running_max[:, 1:]) <= tau_blk[:, 1:]

        element_threshold = m_local[:, :, None] + tau_elem[:, :, None]
        element_pruned = compare_values < element_threshold
        final_pruned = block_pruned[:, :, None] | element_pruned

        y_raw = (fx_blocks - (m_local[:, :, None] << self.cfg.fx_frac_width)) >> (
            self.cfg.fx_frac_width - self.cfg.y_frac_width
        )
        desc_shift, desc_frac = self._build_buffered_descriptor_fast_np(y_raw, final_pruned)

        exp_lut = self._get_exp_lut_np(frac_width)
        delta_lut = self._get_delta_lut_np(frac_width)

        exp_local = np.where(final_pruned, 0, exp_lut[desc_frac] >> desc_shift).astype(np.int32)
        sum_local = exp_local.sum(axis=2).astype(np.int32)

        row_max_final = m_local[:, 0].copy()
        row_sum_final = sum_local[:, 0].copy()
        for block_idx in range(1, block_count):
            live_mask = ~block_pruned[:, block_idx]
            if not np.any(live_mask):
                continue
            use_new_max = live_mask & (m_local[:, block_idx] >= row_max_final)
            if np.any(use_new_max):
                delta = m_local[use_new_max, block_idx] - row_max_final[use_new_max]
                row_sum_final[use_new_max] = (row_sum_final[use_new_max] >> delta) + sum_local[use_new_max, block_idx]
                row_max_final[use_new_max] = m_local[use_new_max, block_idx]
            keep_old_max = live_mask & ~use_new_max
            if np.any(keep_old_max):
                delta = row_max_final[keep_old_max] - m_local[keep_old_max, block_idx]
                row_sum_final[keep_old_max] = row_sum_final[keep_old_max] + (sum_local[keep_old_max, block_idx] >> delta)

        denom_exp = np.zeros(row_count, dtype=np.int32)
        denom_k_raw = np.zeros(row_count, dtype=np.int32)
        positive_sum_mask = row_sum_final > 0
        if np.any(positive_sum_mask):
            small_sum_mask = positive_sum_mask & (row_sum_final < (1 << self.cfg.sum_frac_width))
            denom_exp[small_sum_mask] = -1
            denom_k_raw[small_sum_mask] = (row_sum_final[small_sum_mask] << 1) & ((1 << self.cfg.sum_frac_width) - 1)

            normal_sum_mask = positive_sum_mask & ~small_sum_mask
            if np.any(normal_sum_mask):
                positive_vals = row_sum_final[normal_sum_mask].astype(np.float64)
                highest_bit = np.floor(np.log2(positive_vals)).astype(np.int32)
                denom_exp[normal_sum_mask] = highest_bit - self.cfg.sum_frac_width
                norm_raw = row_sum_final[normal_sum_mask] >> denom_exp[normal_sum_mask]
                denom_k_raw[normal_sum_mask] = norm_raw & ((1 << self.cfg.sum_frac_width) - 1)

        denom_delta_raw = delta_lut[denom_k_raw]
        block_delta = row_max_final[:, None] - m_local

        merged_exp = desc_frac.astype(np.int32) - denom_k_raw[:, None, None] - denom_delta_raw[:, None, None]
        exp_rshift_extra = merged_exp < 0
        lut_in = np.where(
            exp_rshift_extra,
            merged_exp + (1 << self.cfg.sum_frac_width),
            merged_exp & ((1 << self.cfg.sum_frac_width) - 1),
        )
        np.clip(lut_in, 0, (1 << self.cfg.sum_frac_width) - 1, out=lut_in)
        exp_term = exp_lut[lut_in].astype(np.int32)
        final_rshift = desc_shift.astype(np.int32) + block_delta[:, :, None] + denom_exp[:, None, None] + exp_rshift_extra.astype(np.int32)
        shift_left = np.clip(-final_rshift, 0, None).astype(np.int32)
        shift_right = np.clip(final_rshift, 0, None).astype(np.int32)
        shifted_exp = np.where(final_rshift < 0, exp_term << shift_left, exp_term >> shift_right)
        out_raw = np.where(final_pruned, 0, shifted_exp << out_shift).astype(np.int32)

        out_raw &= (1 << self.cfg.out_fmt.total_width) - 1
        approx_probs = out_raw.reshape(row_count, row_depth).astype(np.float32) / out_scale

        prune_stats = {
            "total_elements": int(row_count * row_depth),
            "total_pruned_elements": int(final_pruned.sum()),
            "block_pruned_blocks": int(block_pruned.sum()),
            "total_blocks": int(row_count * block_count),
            "block_pruned_elements": int(block_pruned.sum() * block_size),
            "element_pruned_only": int((element_pruned & ~block_pruned[:, :, None]).sum()),
            "aggressive_blocks": 0 if use_aggressive is None else int(use_aggressive.sum()),
            "conservative_blocks": 0 if use_aggressive is None else int((~use_aggressive).sum()),
        }
        return approx_probs, prune_stats

    def simulate_row_fast(self, row: Sequence[float]) -> Tuple[List[float], List[float], Dict[str, int]]:
        fp_quantized_row, fx_raw_row, _ = self.module1_fp_to_fx(row)
        _, block_max_values = self.module2_block_load_max(fx_raw_row)
        partial_metas, descriptors, y_raw_row, _, element_prune_flags, prune_flags, _ = self.module3_block_prune_y_generate(
            fx_raw_row, block_max_values
        )
        block_flags = [meta.block_pruned for meta in partial_metas]
        block_level_flags = expand_block_flags_to_elements(block_flags, len(descriptors), self.cfg.block_size_effective)
        completed_metas, _ = self.module4_exp_block_reduce(descriptors, partial_metas, y_raw_row)
        row_final_state_token = self.module5_row_online_merge(completed_metas)
        *_, row_normalization_token = self.module6_row_denom_prep(row_final_state_token)
        approx_probs, _ = self.module6_replay_output(descriptors, completed_metas, row_normalization_token)
        prune_stats = {
            "total_elements": len(prune_flags),
            "total_pruned_elements": sum(1 for flag in prune_flags if flag),
            "block_pruned_blocks": sum(1 for flag in block_flags if flag),
            "total_blocks": len(block_flags),
            "block_pruned_elements": sum(1 for flag in block_level_flags if flag),
            "element_pruned_only": sum(
                1
                for elem_pruned, final_pruned, block_pruned in zip(
                    element_prune_flags,
                    prune_flags,
                    block_level_flags,
                )
                if elem_pruned and final_pruned and not block_pruned
            ),
            "aggressive_blocks": (
                sum(1 for meta in partial_metas if meta.c2_count <= self.cfg.adaptive_c2_threshold)
                if self.cfg.adaptive_prune_mode == "block_c2_two_level"
                else 0
            ),
            "conservative_blocks": (
                sum(1 for meta in partial_metas if meta.c2_count > self.cfg.adaptive_c2_threshold)
                if self.cfg.adaptive_prune_mode == "block_c2_two_level"
                else 0
            ),
        }
        return fp_quantized_row, approx_probs, prune_stats


def parse_row(row_text: str) -> List[float]:
    """Parse a command-line row like '1.0,2.0,3.0'."""
    return [float(token.strip()) for token in row_text.split(",") if token.strip()]


def generate_random_row(row_depth: int, seed: int) -> List[float]:
    """Generate a simple random logits row for quick experiments."""
    rng = random.Random(seed)
    return [rng.uniform(-8.0, 8.0) for _ in range(row_depth)]


def generate_random_dataset(
    num_rows: int, row_depth: int, seed: int, value_min: float, value_max: float
) -> List[List[float]]:
    rng = random.Random(seed)
    return [[rng.uniform(value_min, value_max) for _ in range(row_depth)] for _ in range(num_rows)]


def get_workload_spec(workload: str) -> WorkloadSpec:
    specs = {
        "uniform": WorkloadSpec(
            name="uniform",
            default_row_depth=256,
            distribution="uniform",
            value_min=-8.0,
            value_max=8.0,
            normal_mean=0.0,
            normal_std=1.0,
        ),
        "bert-base": WorkloadSpec(
            name="bert-base",
            default_row_depth=128,
            distribution="normal",
            value_min=-6.0,
            value_max=6.0,
            normal_mean=0.0,
            normal_std=1.6,
        ),
        "mobilevit": WorkloadSpec(
            name="mobilevit",
            default_row_depth=1000,
            distribution="normal",
            value_min=-10.0,
            value_max=10.0,
            normal_mean=0.0,
            normal_std=2.8,
        ),
    }
    if workload not in specs:
        raise ValueError(f"unsupported workload: {workload}")
    return specs[workload]


def generate_workload_dataset(
    workload: str,
    num_rows: int,
    row_depth: int,
    seed: int,
    value_min: float,
    value_max: float,
) -> List[List[float]]:
    spec = get_workload_spec(workload)
    rng = random.Random(seed)
    if workload == "uniform":
        return generate_random_dataset(num_rows, row_depth, seed, value_min, value_max)

    rows = []
    for _ in range(num_rows):
        row = []
        for _ in range(row_depth):
            value = rng.gauss(spec.normal_mean, spec.normal_std)
            row.append(max(value_min, min(value_max, value)))
        rows.append(row)
    return rows


def load_rows_from_file(file_path: str) -> List[List[float]]:
    rows: List[List[float]] = []
    with open(file_path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            if text.startswith("["):
                parsed = json.loads(text)
                if not isinstance(parsed, list):
                    raise ValueError(f"{file_path}:{line_no} is not a JSON array")
                row = [float(value) for value in parsed]
            else:
                row = [float(token.strip()) for token in text.split(",") if token.strip()]
            if not row:
                raise ValueError(f"{file_path}:{line_no} produced an empty row")
            rows.append(row)
    if not rows:
        raise ValueError(f"{file_path} did not contain any valid rows")
    row_depth = len(rows[0])
    if any(len(row) != row_depth for row in rows):
        raise ValueError(f"{file_path} contains rows with inconsistent lengths")
    return rows


def parse_shift_terms(terms_text: str) -> Tuple[Tuple[int, int], ...]:
    """Parse shift-add terms.

    Example:
    '+0,+2,+3,+4,+8' -> 1 + 1/4 + 1/8 + 1/16 + 1/256
    """
    parsed = []
    for token in terms_text.split(","):
        token = token.strip()
        if not token:
            continue
        sign = 1
        if token.startswith("-"):
            sign = -1
            token = token[1:]
        elif token.startswith("+"):
            token = token[1:]
        parsed.append((sign, int(token)))
    if not parsed:
        raise ValueError("at least one shift term is required")
    return tuple(parsed)


def result_to_dict(result: SoftmaxRowResult) -> dict:
    data = asdict(result)
    data["descriptors"] = [asdict(desc) for desc in result.descriptors]
    data["row_ctx"] = dict(data["row_final_state_token"])
    data["module6_denom_state"] = dict(data["row_normalization_token"])
    return data


def config_to_dict(cfg: SoftmaxModelConfig) -> dict:
    data = asdict(cfg)
    data["row_len"] = cfg.row_len
    data["lane_count_effective"] = cfg.lane_count_effective
    data["bus_width_effective"] = cfg.bus_width_effective
    data["lane_num"] = cfg.lane_num
    data["block_size_effective"] = cfg.block_size_effective
    data["block_clog2"] = cfg.block_clog2
    data["block_prune_threshold_effective"] = cfg.block_prune_threshold_effective
    if cfg.block_prune_threshold is None:
        data["block_prune_threshold_source"] = {
            "mode": "sum_precision_bound",
            "sum_frac_width": cfg.sum_frac_width,
            "block_clog2": cfg.block_clog2,
        }
    return data


def batch_summary_to_dict(summary: SoftmaxBatchEvalSummary) -> dict:
    return asdict(summary)


def describe_block_prune_threshold(cfg: SoftmaxModelConfig) -> str:
    if cfg.block_prune_threshold is None:
        return (
            f"auto(sum_frac={cfg.sum_frac_width},"
            f"block_clog2={cfg.block_clog2}->{cfg.block_prune_threshold_effective:.1f})"
        )
    return f"{cfg.block_prune_threshold_effective:.1f}"


def describe_elem_prune_policy(cfg: SoftmaxModelConfig) -> str:
    if not cfg.adaptive_prune_enabled:
        return f"fixed({cfg.prune_threshold:.1f})"
    if cfg.adaptive_prune_mode == "block_c2_two_level":
        return (
            f"adaptive_c2(A={cfg.adaptive_c2_threshold}, "
            f"{cfg.adaptive_tau_aggressive_int}/{cfg.adaptive_tau_conservative_int})"
        )
    return cfg.adaptive_prune_mode


def get_precision_profile_overrides(profile: str) -> Dict[str, object]:
    if profile == "custom":
        return {}
    if profile == "doc_adaptive_desc8_q6_special4":
        return {
            "lane_count": 8,
            "block_size": 8,
            "fx_int_width": 6,
            "fx_frac_width": 10,
            "y_int_width": 3,
            "y_frac_width": 6,
            "exp_frac_width": 6,
            "out_frac_width": 15,
            "sum_int_width": 10,
            "sum_frac_width": 6,
            "y_shift_width": 2,
            "y_frac_buf_width": 6,
            "descriptor_mode": "z_code_q6_special_4",
            "sum_use_buffered_descriptor": True,
            "prune_threshold": -4.0,
            "adaptive_prune_mode": "block_c2_two_level",
            "adaptive_c2_threshold": 3,
            "adaptive_tau_aggressive": -2,
            "adaptive_tau_conservative": -4,
            "elem_prune_compare_mode": "floor_int",
            "block_prune_threshold": None,
            "exp_pwl_mode": "DOC",
            "rtl_exact": True,
            "log2e_shift_terms": ((1, 0), (1, 1), (-1, 4)),
        }
    if profile == "doc_adaptive_desc9_q7_special4":
        return {
            "lane_count": 8,
            "block_size": 8,
            "fx_int_width": 6,
            "fx_frac_width": 10,
            "y_int_width": 3,
            "y_frac_width": 7,
            "exp_frac_width": 7,
            "out_frac_width": 15,
            "sum_int_width": 10,
            "sum_frac_width": 7,
            "y_shift_width": 2,
            "y_frac_buf_width": 7,
            "descriptor_mode": "z_code_q7_special_4",
            "sum_use_buffered_descriptor": True,
            "prune_threshold": -4.0,
            "adaptive_prune_mode": "block_c2_two_level",
            "adaptive_c2_threshold": 3,
            "adaptive_tau_aggressive": -2,
            "adaptive_tau_conservative": -4,
            "elem_prune_compare_mode": "floor_int",
            "block_prune_threshold": None,
            "exp_pwl_mode": "DOC",
            "rtl_exact": True,
            "log2e_shift_terms": ((1, 0), (1, 1), (-1, 4)),
        }
    if profile == "doc_adaptive_desc9_q7_special4_block4":
        return {
            "lane_count": 4,
            "block_size": 4,
            "fx_int_width": 6,
            "fx_frac_width": 10,
            "y_int_width": 3,
            "y_frac_width": 7,
            "exp_frac_width": 7,
            "out_frac_width": 15,
            "sum_int_width": 10,
            "sum_frac_width": 7,
            "y_shift_width": 2,
            "y_frac_buf_width": 7,
            "descriptor_mode": "z_code_q7_special_4",
            "sum_use_buffered_descriptor": True,
            "prune_threshold": -4.0,
            "adaptive_prune_mode": "block_c2_two_level",
            "adaptive_c2_threshold": 3,
            "adaptive_tau_aggressive": -2,
            "adaptive_tau_conservative": -4,
            "elem_prune_compare_mode": "floor_int",
            "block_prune_threshold": None,
            "exp_pwl_mode": "DOC",
            "rtl_exact": True,
            "log2e_shift_terms": ((1, 0), (1, 1), (-1, 4)),
        }
    raise ValueError(f"unsupported precision profile: {profile}")


def build_config_from_args(args: argparse.Namespace, profile: str) -> SoftmaxModelConfig:
    cfg_kwargs = {
        "input_format": args.input_format,
        "bus_width": args.bus_width,
        "elem_width": args.elem_width,
        "lane_count": args.lane_count,
        "row_depth": args.row_depth,
        "block_size": args.block_size,
        "fx_int_width": args.fx_int_width,
        "fx_frac_width": args.fx_frac_width,
        "y_int_width": args.y_int_width,
        "y_frac_width": args.y_frac_width,
        "exp_frac_width": args.exp_frac_width,
        "out_frac_width": args.out_frac_width,
        "sum_int_width": args.sum_int_width,
        "sum_frac_width": args.sum_frac_width,
        "y_shift_width": args.y_shift_width,
        "y_frac_buf_width": args.y_frac_buf_width,
        "descriptor_mode": "shift_frac",
        "sum_use_buffered_descriptor": args.sum_use_buffered_descriptor,
        "prune_threshold": args.prune_threshold,
        "adaptive_prune_mode": args.adaptive_prune_mode,
        "adaptive_c2_threshold": args.adaptive_c2_threshold,
        "adaptive_tau_aggressive": args.adaptive_tau_aggressive,
        "adaptive_tau_conservative": args.adaptive_tau_conservative,
        "elem_prune_compare_mode": args.elem_prune_compare_mode,
        "block_prune_enabled": not args.disable_block_prune,
        "block_prune_threshold": args.block_prune_threshold,
        "exp_pwl_mode": args.exp_pwl_mode,
        "log2e_shift_terms": parse_shift_terms(args.log2e_shifts),
    }
    cfg_kwargs.update(get_precision_profile_overrides(profile))
    return SoftmaxModelConfig(**cfg_kwargs)


def format_float_list(values: Sequence[float], digits: int = 8) -> str:
    return ",".join(f"{value:.{digits}f}" for value in values)


def format_topk_counts(topk_counts: Dict[str, int]) -> str:
    return ", ".join(f"top{k}={count}" for k, count in sorted(topk_counts.items(), key=lambda item: int(item[0])))


def format_topk_rates(topk_rates: Dict[str, float]) -> str:
    return ", ".join(f"top{k}={rate:.6f}" for k, rate in sorted(topk_rates.items(), key=lambda item: int(item[0])))


def argmax_index(values: Sequence[float]) -> int:
    if not values:
        raise ValueError("values must not be empty")
    best_idx = 0
    best_value = values[0]
    for idx, value in enumerate(values[1:], start=1):
        if value > best_value:
            best_idx = idx
            best_value = value
    return best_idx


def summarize_profile(cfg: SoftmaxModelConfig) -> str:
    shift_text = ",".join(f"{'+' if sign >= 0 else '-'}{shift}" for sign, shift in cfg.log2e_shift_terms)
    return (
        f"fx=Q{cfg.fx_int_width}.{cfg.fx_frac_width}, "
        f"y=Q{cfg.y_int_width}.{cfg.y_frac_width}, "
        f"lane={cfg.lane_count_effective}, "
        f"row_len={cfg.row_len}, "
        f"block={cfg.block_size_effective}, "
        f"exp_frac={cfg.exp_frac_width}, "
        f"out_frac={cfg.out_frac_width}, sum_frac={cfg.sum_frac_width}, "
        f"y_buf_frac={cfg.y_frac_buf_width}, exp_pwl={cfg.exp_pwl_mode}, "
        f"prune={describe_elem_prune_policy(cfg)}, elem_prune_mode={cfg.elem_prune_compare_mode}, "
        f"block_prune_threshold={describe_block_prune_threshold(cfg)}, "
        f"block_prune={'on' if cfg.block_prune_enabled else 'off'}, log2e={shift_text}"
    )


def print_output_field_guide() -> None:
    print("输出字段说明")
    print(
        "profile            : 当前精度档位，"
        "doc_adaptive_desc8_q6_special4 / "
        "doc_adaptive_desc9_q7_special4 / "
        "doc_adaptive_desc9_q7_special4_block4 / custom"
    )
    print("input_format       : 输入格式，fp16 或 bf16")
    print("lane_count         : 内部并行 lane 数，是当前文档的核心并行参数")
    print("bus_width          : 外部接口总线位宽；缺省时按 lane_count * elem_width 推导")
    print("block_size         : 一个 block 包含的元素数；缺省时等于 lane_count")
    print("row_len            : 一行 softmax 输入里包含的元素个数")
    print("fx_format          : 模块1输出的 base-2 域内部定点格式 Qm.n")
    print("y_format           : 模块3使用的 y 域定点格式 Qm.n")
    print("exp_pwl_mode       : 2^v 近似模式；当前仅保留文档冻结的 DOC=整段割线+3/16三角补偿")
    print("prune_threshold    : 固定元素剪枝门限；当 adaptive_prune_mode=fixed 时生效")
    print("adaptive_prune_mode: 元素剪枝策略；fixed 或 block_c2_two_level")
    print("adaptive_c2_threshold: 两段自适应的 A 值，当前 block 满足 c2<=A 时切到激进档")
    print("elem_prune_mode    : 元素级剪枝判定模式；默认 floor_int，对应补码高位直接比较")
    print("block_prune        : block 级剪枝开关")
    print("block_prune_threshold: block 级剪枝门限；缺省时自动按 -(sum_frac_width + clog2(block_size)) 派生")
    print("log2e_shift_terms  : 用移位加减近似 log2(e) 的项")
    print("row_ctx            : 模块5写出的最小行级上下文 {blk_total, m_global_final, sum_global_final_raw}")
    print("normalized_row_sum : 模块6行启动阶段对分母规格化后的 M * 2^E")
    print("denom_k / denom_delta: 模块6本地缓存的分母补偿量")
    print("max_abs_error      : 近似 softmax 与精确 softmax 的最大绝对误差")
    print("mean_abs_error     : 行内平均绝对误差，可视作单行 MAE")
    print("mse                : 行内均方误差")
    print("approx_probs       : 当前硬件近似模型输出的 softmax 概率")
    print("reference_probs    : 精确浮点 softmax 概率，用作基线")


def print_batch_field_guide() -> None:
    print("批量评估字段说明")
    print("profile            : 当前评估使用的精度档位")
    print("num_rows           : 参与统计的输入行数量")
    print("row_len            : 每一行的元素个数，也就是一次 softmax 的类别数/长度")
    print("seed               : 随机数据种子")
    print("value_range        : 随机生成输入数据的取值范围")
    print("topk_match_counts  : approx 与 reference 的 top-k 排序索引完全一致的行数")
    print("topk_match_rates   : 上述 top-k 一致率")
    print("top1_match_count   : 近似 softmax 与精确 softmax 的 top1 一致行数")
    print("top1_match_rate    : top1 一致率，可理解为这里的“分类准确率”")
    print("mae               : 所有概率点的平均绝对误差")
    print("mse               : 所有概率点的平均均方误差")
    print("avg_max_abs_error  : 对每一行 max_abs_error 再求平均")
    print("avg_mean_abs_error : 对每一行 mean_abs_error 再求平均")
    print("global_max_abs_err : 所有行里出现过的最大绝对误差")
    print("avg_l1_error       : 每一行所有概率绝对误差求和，再对所有行求平均")
    print("avg_prob_sum       : 近似 softmax 概率和的平均值，理想情况下接近 1")
    print("avg_prob_sum_error : 概率和偏离 1 的平均绝对值")
    print("block_pruned_*     : 整块剪枝命中的 block/元素统计")
    print("说明               : 当前“准确率”指 top1_match_rate，不是 top-k 全集准确率")


def print_result_summary(profile: str, cfg: SoftmaxModelConfig, result: SoftmaxRowResult) -> None:
    shift_text = ",".join(f"{'+' if sign >= 0 else '-'}{shift}" for sign, shift in cfg.log2e_shift_terms)
    print(f"Softmax edge model summary [{profile}]")
    print(f"profile           : {profile}")
    print(f"input_format      : {cfg.input_format}")
    print(f"lane_count        : {cfg.lane_count_effective}")
    print(f"bus_width         : {cfg.bus_width_effective}")
    print(f"block_size        : {cfg.block_size_effective}")
    print(f"row_len           : {cfg.row_len}")
    print(f"fx_format         : Q{cfg.fx_int_width}.{cfg.fx_frac_width}")
    print(f"y_format          : Q{cfg.y_int_width}.{cfg.y_frac_width}")
    print(f"exp_pwl_mode      : {cfg.exp_pwl_mode}")
    print(f"prune_threshold   : {describe_elem_prune_policy(cfg)}")
    print(f"adaptive_prune_mode: {cfg.adaptive_prune_mode}")
    if cfg.adaptive_prune_enabled:
        print(
            f"adaptive_c2_threshold: {cfg.adaptive_c2_threshold} "
            f"(aggressive={cfg.adaptive_tau_aggressive_int}, conservative={cfg.adaptive_tau_conservative_int})"
        )
    print(f"elem_prune_mode   : {cfg.elem_prune_compare_mode}")
    print(f"block_prune_threshold: {describe_block_prune_threshold(cfg)}")
    print(f"block_prune       : {'enabled' if cfg.block_prune_enabled else 'disabled'}")
    print(f"log2e_shift_terms : {shift_text}")
    print(f"row_max_value_final: {result.row_max_value_final}")
    print(f"row_sum_final     : {result.row_sum_final:.8f}")
    print(
        "row_ctx          : "
        f"{{blk_total={result.row_final_state_token.block_count}, "
        f"m_global_final={result.row_final_state_token.row_max_value_final}, "
        f"sum_global_final_raw={result.row_final_state_token.row_sum_final_raw}}}"
    )
    print(f"normalized_row_sum_mantissa: {result.normalized_row_sum_mantissa:.8f}")
    print(f"normalized_row_sum_exponent: {result.normalized_row_sum_exponent}")
    print(f"denom_k           : {result.denom_k:.8f}")
    print(f"denom_delta       : {result.denom_delta:.8f}")
    print(f"max_abs_error     : {result.max_abs_error:.8e}")
    print(f"mean_abs_error    : {result.mean_abs_error:.8e}")
    print(f"mse               : {result.mse:.8e}")
    print(f"block_prune_flags : {','.join('1' if flag else '0' for flag in result.block_prune_flags)}")
    print("approx_probs      :", format_float_list(result.approx_probs))
    print("reference_probs   :", format_float_list(result.reference_probs))


def expand_block_flags_to_elements(block_flags: Sequence[bool], total_elems: int, block_size: int) -> List[bool]:
    expanded: List[bool] = []
    for block_idx, block_flag in enumerate(block_flags):
        remaining = total_elems - block_idx * block_size
        if remaining <= 0:
            break
        expanded.extend([block_flag] * min(block_size, remaining))
    return expanded


def _create_batch_accumulator(topk_values: Sequence[int]) -> dict:
    return {
        "topk_match_counts": {str(k): 0 for k in topk_values},
        "top1_match_count": 0,
        "sum_abs_error": 0.0,
        "sum_sq_error": 0.0,
        "num_values": 0,
        "sum_row_max_abs_error": 0.0,
        "sum_row_mean_abs_error": 0.0,
        "global_max_abs_error": 0.0,
        "sum_l1_error": 0.0,
        "sum_prob_sum": 0.0,
        "sum_prob_sum_error": 0.0,
        "sum_kl_divergence": 0.0,
        "block_pruned_block_count": 0,
        "block_pruned_element_count": 0,
        "element_pruned_only_count": 0,
        "total_pruned_element_count": 0,
        "num_rows": 0,
    }


def _accumulate_result(accum: dict, cfg: SoftmaxModelConfig, result: SoftmaxRowResult, topk_values: Sequence[int]) -> None:
    approx_top1 = argmax_index(result.approx_probs)
    ref_top1 = argmax_index(result.reference_probs)
    accum["top1_match_count"] += int(approx_top1 == ref_top1)
    ref_topk = {k: topk_indices(result.reference_probs, k) for k in topk_values}
    approx_topk = {k: topk_indices(result.approx_probs, k) for k in topk_values}
    for k in topk_values:
        accum["topk_match_counts"][str(k)] += int(approx_topk[k] == ref_topk[k])
    abs_errors = [abs(approx - ref) for approx, ref in zip(result.approx_probs, result.reference_probs)]
    sq_errors = [(approx - ref) ** 2 for approx, ref in zip(result.approx_probs, result.reference_probs)]
    accum["sum_abs_error"] += sum(abs_errors)
    accum["sum_sq_error"] += sum(sq_errors)
    accum["num_values"] += len(result.approx_probs)
    accum["sum_row_max_abs_error"] += result.max_abs_error
    accum["sum_row_mean_abs_error"] += result.mean_abs_error
    accum["global_max_abs_error"] = max(accum["global_max_abs_error"], result.max_abs_error)
    accum["sum_l1_error"] += sum(abs_errors)
    prob_sum = sum(result.approx_probs)
    accum["sum_prob_sum"] += prob_sum
    accum["sum_prob_sum_error"] += abs(prob_sum - 1.0)
    accum["sum_kl_divergence"] += result.kl_divergence
    accum["block_pruned_block_count"] += sum(1 for flag in result.block_prune_flags if flag)
    block_level_flags = expand_block_flags_to_elements(result.block_prune_flags, len(result.prune_flags), cfg.block_size_effective)
    accum["block_pruned_element_count"] += sum(1 for block_pruned in block_level_flags if block_pruned)
    accum["element_pruned_only_count"] += sum(
        1
        for elem_pruned, final_pruned, block_pruned in zip(
            result.element_prune_flags,
            result.prune_flags,
            block_level_flags,
        )
        if elem_pruned and final_pruned and not block_pruned
    )
    accum["total_pruned_element_count"] += sum(1 for flag in result.prune_flags if flag)
    accum["num_rows"] += 1


def _merge_accumulators(dst: dict, src: dict) -> None:
    for key, value in src["topk_match_counts"].items():
        dst["topk_match_counts"][key] += value
    for key in (
        "top1_match_count",
        "sum_abs_error",
        "sum_sq_error",
        "num_values",
        "sum_row_max_abs_error",
        "sum_row_mean_abs_error",
        "sum_l1_error",
        "sum_prob_sum",
        "sum_prob_sum_error",
        "sum_kl_divergence",
        "block_pruned_block_count",
        "block_pruned_element_count",
        "element_pruned_only_count",
        "total_pruned_element_count",
        "num_rows",
    ):
        dst[key] += src[key]
    dst["global_max_abs_error"] = max(dst["global_max_abs_error"], src["global_max_abs_error"])


def _evaluate_rows_chunk(cfg: SoftmaxModelConfig, rows: Sequence[Sequence[float]], topk_values: Sequence[int]) -> dict:
    model = SoftmaxEdgeModel(cfg)
    accum = _create_batch_accumulator(topk_values)
    for row in rows:
        _accumulate_result(accum, cfg, model.simulate_row(row), topk_values)
    return accum


def _split_row_chunks(rows: Sequence[Sequence[float]], num_chunks: int) -> List[List[List[float]]]:
    num_chunks = max(1, min(num_chunks, len(rows)))
    chunk_size = math.ceil(len(rows) / num_chunks)
    chunks: List[List[List[float]]] = []
    for idx in range(0, len(rows), chunk_size):
        chunks.append([list(row) for row in rows[idx : idx + chunk_size]])
    return chunks


def evaluate_rows(
    profile: str,
    cfg: SoftmaxModelConfig,
    rows: Sequence[Sequence[float]],
    seed: int,
    value_min: float,
    value_max: float,
    topk_values: Sequence[int],
    num_workers: int = 1,
) -> SoftmaxBatchEvalSummary:
    num_rows = len(rows)
    if num_rows == 0:
        raise ValueError("rows must not be empty")
    accum = _create_batch_accumulator(topk_values)
    if num_workers > 1 and num_rows > 1:
        chunks = _split_row_chunks(rows, num_workers)
        max_workers = min(num_workers, len(chunks))
        try:
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(_evaluate_rows_chunk, cfg, chunk, tuple(topk_values)) for chunk in chunks]
                for future in concurrent.futures.as_completed(futures):
                    _merge_accumulators(accum, future.result())
        except (PermissionError, OSError):
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(_evaluate_rows_chunk, cfg, chunk, tuple(topk_values)) for chunk in chunks]
                for future in concurrent.futures.as_completed(futures):
                    _merge_accumulators(accum, future.result())
    else:
        model = SoftmaxEdgeModel(cfg)
        for row in rows:
            _accumulate_result(accum, cfg, model.simulate_row(row), topk_values)

    if accum["num_values"] == 0:
        raise ValueError("rows must contain at least one value")
    topk_match_rates = {key: value / num_rows for key, value in accum["topk_match_counts"].items()}
    return SoftmaxBatchEvalSummary(
        profile=profile,
        num_rows=num_rows,
        row_depth=len(rows[0]),
        seed=seed,
        value_min=value_min,
        value_max=value_max,
        topk_values=list(topk_values),
        topk_match_counts=accum["topk_match_counts"],
        topk_match_rates=topk_match_rates,
        top1_match_count=accum["top1_match_count"],
        top1_match_rate=accum["top1_match_count"] / num_rows,
        mae=accum["sum_abs_error"] / accum["num_values"],
        mse=accum["sum_sq_error"] / accum["num_values"],
        avg_max_abs_error=accum["sum_row_max_abs_error"] / num_rows,
        avg_mean_abs_error=accum["sum_row_mean_abs_error"] / num_rows,
        global_max_abs_error=accum["global_max_abs_error"],
        avg_l1_error=accum["sum_l1_error"] / num_rows,
        avg_prob_sum=accum["sum_prob_sum"] / num_rows,
        avg_prob_sum_error=accum["sum_prob_sum_error"] / num_rows,
        block_pruned_block_count=accum["block_pruned_block_count"],
        block_pruned_element_count=accum["block_pruned_element_count"],
        element_pruned_only_count=accum["element_pruned_only_count"],
        total_pruned_element_count=accum["total_pruned_element_count"],
        avg_kl_divergence=accum["sum_kl_divergence"] / num_rows,
        avg_perplexity=math.exp(accum["sum_kl_divergence"] / num_rows),
    )


def print_batch_summary(summary: SoftmaxBatchEvalSummary, cfg: SoftmaxModelConfig) -> None:
    print(f"Softmax batch evaluation [{summary.profile}]")
    print(f"profile           : {summary.profile}")
    print(f"profile_summary   : {summarize_profile(cfg)}")
    print(f"num_rows          : {summary.num_rows}")
    print(f"row_len           : {summary.row_depth}")
    print(f"seed              : {summary.seed}")
    print(f"value_range       : [{summary.value_min:.4f}, {summary.value_max:.4f}]")
    print(f"topk_match_counts : {format_topk_counts(summary.topk_match_counts)}")
    print(f"topk_match_rates  : {format_topk_rates(summary.topk_match_rates)}")
    print(f"top1_match_count  : {summary.top1_match_count}")
    print(f"top1_match_rate   : {summary.top1_match_rate:.6f}")
    print(f"mae               : {summary.mae:.8e}")
    print(f"mse               : {summary.mse:.8e}")
    print(f"avg_max_abs_error : {summary.avg_max_abs_error:.8e}")
    print(f"avg_mean_abs_err  : {summary.avg_mean_abs_error:.8e}")
    print(f"global_max_abs_err: {summary.global_max_abs_error:.8e}")
    print(f"avg_l1_error      : {summary.avg_l1_error:.8e}")
    print(f"avg_prob_sum      : {summary.avg_prob_sum:.8f}")
    print(f"avg_prob_sum_error: {summary.avg_prob_sum_error:.8e}")
    print(f"block_pruned_blocks: {summary.block_pruned_block_count}")
    print(f"block_pruned_elements: {summary.block_pruned_element_count}")
    print(f"element_pruned_only: {summary.element_pruned_only_count}")
    print(f"total_pruned_elements: {summary.total_pruned_element_count}")
    print(f"avg_kl_divergence : {summary.avg_kl_divergence:.8e}")
    print(f"avg_perplexity    : {summary.avg_perplexity:.6f}")


def build_batch_compare_dict(profile_runs: Sequence[Tuple[str, SoftmaxModelConfig, SoftmaxBatchEvalSummary]]) -> dict:
    payload = {
        profile: {
            "config": config_to_dict(cfg),
            "summary": batch_summary_to_dict(summary),
        }
        for profile, cfg, summary in profile_runs
    }
    payload["comparison"] = {
        "profiles": [profile for profile, _, _ in profile_runs],
        "profile_summaries": {
            profile: summarize_profile(cfg) for profile, cfg, _ in profile_runs
        },
    }
    return payload


def print_batch_compare_summary(profile_runs: Sequence[Tuple[str, SoftmaxModelConfig, SoftmaxBatchEvalSummary]]) -> None:
    first_profile, _, first_summary = profile_runs[0]
    del first_profile
    print("Softmax batch evaluation comparison")
    print(f"num_rows          : {first_summary.num_rows}")
    print(f"row_len           : {first_summary.row_depth}")
    print(f"seed              : {first_summary.seed}")
    print(f"value_range       : [{first_summary.value_min:.4f}, {first_summary.value_max:.4f}]")
    for profile, cfg, summary in profile_runs:
        print(f"{profile}_profile     : {summarize_profile(cfg)}")
        print(f"{profile}_topk_rate   : {format_topk_rates(summary.topk_match_rates)}")
        print(f"{profile}_mae         : {summary.mae:.8e}")
        print(f"{profile}_mse         : {summary.mse:.8e}")
        print(f"{profile}_global_max  : {summary.global_max_abs_error:.8e}")
        print(f"{profile}_kl_div      : {summary.avg_kl_divergence:.8e}")
        print(f"{profile}_ppl         : {summary.avg_perplexity:.6f}")
        print(f"{profile}_psum_err    : {summary.avg_prob_sum_error:.8e}")


def build_compare_dict(
    row: Sequence[float],
    profile_runs: Sequence[Tuple[str, SoftmaxModelConfig, SoftmaxRowResult]],
    topk_values: Sequence[int],
) -> dict:
    payload = {
        "input_row": list(row),
        "topk_values": list(topk_values),
    }
    for profile, cfg, result in profile_runs:
        payload[profile] = {
            "config": config_to_dict(cfg),
            "result": result_to_dict(result),
        }
    payload["comparison"] = {
        "profiles": [profile for profile, _, _ in profile_runs],
        "profile_summaries": {
            profile: summarize_profile(cfg) for profile, cfg, _ in profile_runs
        },
    }
    return payload


def print_compare_summary(
    row: Sequence[float],
    profile_runs: Sequence[Tuple[str, SoftmaxModelConfig, SoftmaxRowResult]],
    topk_values: Sequence[int],
) -> None:
    print("Softmax precision comparison")
    print("input_row         :", format_float_list(row, digits=4))
    reference_probs = profile_runs[0][2].reference_probs
    for profile, cfg, result in profile_runs:
        topk_matches = {
            str(k): int(topk_indices(result.approx_probs, k) == topk_indices(reference_probs, k))
            for k in topk_values
        }
        print(f"{profile}_profile     : {summarize_profile(cfg)}")
        print(f"{profile}_topk_match  : {format_topk_counts(topk_matches)}")
        print(f"{profile}_mae         : {result.mean_abs_error:.8e}")
        print(f"{profile}_mse         : {result.mse:.8e}")
        print(f"{profile}_max_abs_err : {result.max_abs_error:.8e}")
        print(f"{profile}_kl_div      : {result.kl_divergence:.8e}")
        print(f"{profile}_probs       : {format_float_list(result.approx_probs)}")
    print("reference_probs   :", format_float_list(reference_probs))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Edge softmax accelerator Python model")
    parser.add_argument("--row", type=str, help="comma-separated input row values")
    parser.add_argument("--row-len", "--row-depth", dest="row_depth", type=int, default=256, help="row length when generating random data")
    parser.add_argument("--seed", type=int, default=7, help="seed for random input generation")
    parser.add_argument(
        "--workload",
        choices=("uniform", "bert-base", "mobilevit"),
        default="uniform",
        help="随机数据生成时参考的 softmax workload 类型",
    )
    parser.add_argument(
        "--dataset-file",
        type=str,
        help="从文件读取真实或预先导出的 logits；每行支持 CSV 或 JSON 数组",
    )
    parser.add_argument(
        "--eval-rows",
        type=int,
        default=0,
        help="批量评估的随机输入行数量；为 0 时表示只跑单行",
    )
    parser.add_argument(
        "--eval-min",
        type=float,
        default=-8.0,
        help="批量评估随机输入的最小值",
    )
    parser.add_argument(
        "--eval-max",
        type=float,
        default=8.0,
        help="批量评估随机输入的最大值",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="批量评估时的并行进程数；1 表示串行",
    )
    parser.add_argument(
        "--precision-profile",
        choices=(
            "custom",
            "doc_adaptive_desc8_q6_special4",
            "doc_adaptive_desc9_q7_special4",
            "doc_adaptive_desc9_q7_special4_block4",
        ),
        default="custom",
        help="选择单次运行的精度档位；custom 表示完全使用命令行参数",
    )
    parser.add_argument(
        "--compare-precision",
        action="store_true",
        help="同时运行默认的 doc_adaptive_desc8_q6_special4 与 custom，并打印对比结果",
    )
    parser.add_argument(
        "--compare-profiles",
        type=str,
        default="",
        help="逗号分隔的 profile 列表；为空时 --compare-precision 默认使用 doc_adaptive_desc8_q6_special4,custom",
    )
    parser.add_argument(
        "--explain-output",
        action="store_true",
        help="打印摘要字段含义，便于理解输出内容",
    )
    parser.add_argument("--input-format", choices=("fp16", "bf16"), default="fp16")
    parser.add_argument(
        "--bus-width",
        type=int,
        default=None,
        help="external interface bus width; omitted means lane_count * elem_width",
    )
    parser.add_argument("--elem-width", type=int, default=16)
    parser.add_argument("--lane-count", type=int, default=8, help="internal parallel lane count used by the optimized design")
    parser.add_argument(
        "--block-size",
        type=int,
        default=None,
        help="block size; default follows lane_count to match the optimized design doc",
    )
    parser.add_argument("--fx-int-width", type=int, default=6)
    parser.add_argument("--fx-frac-width", type=int, default=10)
    parser.add_argument("--y-int-width", type=int, default=6)
    parser.add_argument("--y-frac-width", type=int, default=10)
    parser.add_argument("--exp-frac-width", type=int, default=10)
    parser.add_argument("--out-frac-width", type=int, default=10)
    parser.add_argument("--sum-int-width", type=int, default=10)
    parser.add_argument("--sum-frac-width", type=int, default=10)
    parser.add_argument("--y-shift-width", type=int, default=4)
    parser.add_argument("--y-frac-buf-width", type=int, default=10)
    parser.add_argument(
        "--sum-use-buffered-descriptor",
        action="store_true",
        help="模块4分母求和路径直接消费压缩后的 descriptor，而不是回退到未压缩 y",
    )
    parser.add_argument("--prune-threshold", type=float, default=-10.0)
    parser.add_argument(
        "--adaptive-prune-mode",
        choices=("fixed", "block_c2_two_level"),
        default="fixed",
        help="元素剪枝策略；fixed 为固定门限，block_c2_two_level 为按 c2 做 block-local 两段自适应",
    )
    parser.add_argument(
        "--adaptive-c2-threshold",
        type=int,
        default=3,
        help="两段自适应中的 A 值；当 c2<=A 时切到 aggressive 门限",
    )
    parser.add_argument(
        "--adaptive-tau-aggressive",
        type=int,
        default=-2,
        help="两段自适应中的激进元素门限",
    )
    parser.add_argument(
        "--adaptive-tau-conservative",
        type=int,
        default=-4,
        help="两段自适应中的保守元素门限",
    )
    parser.add_argument(
        "--elem-prune-compare-mode",
        choices=("full_y", "floor_int", "trunc_int"),
        default="floor_int",
        help="元素剪枝比较方式；默认 floor_int，对应补码定点高位直接比较，硬件最简单",
    )
    parser.add_argument(
        "--block-prune-threshold",
        type=float,
        default=None,
        help="显式指定 block 门限；省略时自动按 -(sum_frac_width + clog2(block_size)) 派生",
    )
    parser.add_argument("--disable-block-prune", action="store_true")
    parser.add_argument("--exp-pwl-mode", choices=("DOC",), default="DOC")
    parser.add_argument(
        "--topk-values",
        type=str,
        default="1,3,5",
        help="逗号分隔 top-k 列表，例如 1,3,5",
    )
    parser.add_argument(
        "--log2e-shifts",
        type=str,
        default="+0,+1,-4",
        help="comma-separated signed shift terms for log2(e), example '+0,+1,-4'",
    )
    parser.add_argument("--json", action="store_true", help="dump full JSON result")
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    topk_values = parse_topk_values(args.topk_values)
    compare_profiles = (
        tuple(profile.strip() for profile in args.compare_profiles.split(",") if profile.strip())
        if args.compare_profiles
        else DEFAULT_COMPARE_PROFILES
    )
    compare_requested = args.compare_precision or bool(args.compare_profiles)
    if args.dataset_file:
        rows_from_file = load_rows_from_file(args.dataset_file)
        row = rows_from_file[0]
        args.row_depth = len(row)
    elif args.row:
        row = parse_row(args.row)
        args.row_depth = len(row)
    else:
        workload_spec = get_workload_spec(args.workload)
        if args.row_depth == parser.get_default("row_depth"):
            args.row_depth = workload_spec.default_row_depth
        row = generate_workload_dataset(
            args.workload,
            1,
            args.row_depth,
            args.seed,
            args.eval_min if args.eval_min != parser.get_default("eval_min") else workload_spec.value_min,
            args.eval_max if args.eval_max != parser.get_default("eval_max") else workload_spec.value_max,
        )[0]
    args.row_depth = len(row)

    if args.eval_rows > 0:
        if args.dataset_file:
            rows = rows_from_file
            args.eval_rows = len(rows)
            args.row_depth = len(rows[0])
            eval_min = min(min(row_data) for row_data in rows)
            eval_max = max(max(row_data) for row_data in rows)
        else:
            workload_spec = get_workload_spec(args.workload)
            eval_min = args.eval_min if args.eval_min != parser.get_default("eval_min") else workload_spec.value_min
            eval_max = args.eval_max if args.eval_max != parser.get_default("eval_max") else workload_spec.value_max
            rows = generate_workload_dataset(args.workload, args.eval_rows, args.row_depth, args.seed, eval_min, eval_max)

        if compare_requested:
            profile_runs = []
            for profile in compare_profiles:
                cfg = build_config_from_args(args, profile)
                summary = evaluate_rows(profile, cfg, rows, args.seed, eval_min, eval_max, topk_values, num_workers=args.num_workers)
                profile_runs.append((profile, cfg, summary))

            if args.json:
                print(json.dumps(build_batch_compare_dict(profile_runs), indent=2))
            else:
                print_batch_compare_summary(profile_runs)
                if args.explain_output:
                    print()
                    print_batch_field_guide()
            return 0

        cfg = build_config_from_args(args, args.precision_profile)
        summary = evaluate_rows(
            args.precision_profile,
            cfg,
            rows,
            args.seed,
            eval_min,
            eval_max,
            topk_values,
            num_workers=args.num_workers,
        )
        if args.json:
            print(
                json.dumps(
                    {
                        "profile": args.precision_profile,
                        "config": config_to_dict(cfg),
                        "summary": batch_summary_to_dict(summary),
                    },
                    indent=2,
                )
            )
        else:
            print_batch_summary(summary, cfg)
            if args.explain_output:
                print()
                print_batch_field_guide()
        return 0

    if compare_requested:
        profile_runs = []
        for profile in compare_profiles:
            cfg = build_config_from_args(args, profile)
            result = SoftmaxEdgeModel(cfg).simulate_row(row)
            profile_runs.append((profile, cfg, result))

        if args.json:
            print(json.dumps(build_compare_dict(row, profile_runs, topk_values), indent=2))
        else:
            print_compare_summary(row, profile_runs, topk_values)
            if args.explain_output:
                print()
                print_output_field_guide()
        return 0

    cfg = build_config_from_args(args, args.precision_profile)
    result = SoftmaxEdgeModel(cfg).simulate_row(row)

    if args.json:
        print(json.dumps({"profile": args.precision_profile, "config": config_to_dict(cfg), "result": result_to_dict(result)}, indent=2))
        return 0

    print_result_summary(args.precision_profile, cfg, result)
    if args.explain_output:
        print()
        print_output_field_guide()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
