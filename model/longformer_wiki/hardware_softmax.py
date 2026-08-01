"""Torch softmax patch backed by the RTL-aligned hardware reference model."""

from __future__ import annotations

import contextlib
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F

try:
    from .experiment_paths import configure_environment
except ImportError:  # pragma: no cover - supports direct script execution
    from experiment_paths import configure_environment


configure_environment()

try:
    from llama7.softmax_edge_model import (
        SoftmaxEdgeModel,
        SoftmaxModelConfig,
        get_precision_profile_overrides,
    )
except ModuleNotFoundError:  # pragma: no cover - supports GitHub repo layout
    try:
        from softmax_edge_model import (
            SoftmaxEdgeModel,
            SoftmaxModelConfig,
            get_precision_profile_overrides,
        )
    except ModuleNotFoundError as second_exc:  # pragma: no cover - only hit if repo layout is broken
        SoftmaxEdgeModel = None  # type: ignore[assignment]
        SoftmaxModelConfig = None  # type: ignore[assignment]
        get_precision_profile_overrides = None  # type: ignore[assignment]
        SOFTMAX_IMPORT_ERROR = second_exc
    else:
        SOFTMAX_IMPORT_ERROR = None
else:
    SOFTMAX_IMPORT_ERROR = None


DEFAULT_SOFTMAX_PROFILE = "HSSS-Softmax-block8"


class HardwareSoftmaxRuntime:
    """Adapt ``SoftmaxEdgeModel`` to ``torch.softmax`` and ``F.softmax`` call shapes."""

    def __init__(
        self,
        profile: str,
        input_format: str = "fp16",
        fallback_softmax=None,
    ):
        self.profile = profile
        self.input_format = input_format
        self.fallback_softmax = fallback_softmax or F.softmax
        self.model_cache: Dict[int, SoftmaxEdgeModel] = {}
        self.stats = self._empty_stats()

    @staticmethod
    def _empty_stats() -> dict:
        return {
            "calls": 0,
            "total_rows": 0,
            "total_elements": 0,
            "total_pruned_elements": 0,
            "block_pruned_blocks": 0,
            "total_blocks": 0,
            "block_pruned_elements": 0,
            "element_pruned_only": 0,
            "aggressive_blocks": 0,
            "conservative_blocks": 0,
            "padded_elements": 0,
        }

    def _get_model(self, row_depth: int) -> SoftmaxEdgeModel:
        model = self.model_cache.get(row_depth)
        if model is not None:
            return model
        if SoftmaxEdgeModel is None or SoftmaxModelConfig is None or get_precision_profile_overrides is None:
            raise RuntimeError(f"softmax model import failed: {SOFTMAX_IMPORT_ERROR}")

        cfg_kwargs = {
            "input_format": self.input_format,
            "row_depth": row_depth,
        }
        cfg_kwargs.update(get_precision_profile_overrides(self.profile))
        model = SoftmaxEdgeModel(SoftmaxModelConfig(**cfg_kwargs))
        self.model_cache[row_depth] = model
        return model

    def _aggregate_stats(self, prune_stats: dict, padded_elements: int, row_count: int) -> None:
        self.stats["calls"] += 1
        self.stats["total_rows"] += row_count
        self.stats["total_elements"] += int(prune_stats.get("total_elements", 0))
        self.stats["total_pruned_elements"] += int(prune_stats.get("total_pruned_elements", 0))
        self.stats["block_pruned_blocks"] += int(prune_stats.get("block_pruned_blocks", 0))
        self.stats["total_blocks"] += int(prune_stats.get("total_blocks", 0))
        self.stats["block_pruned_elements"] += int(prune_stats.get("block_pruned_elements", 0))
        self.stats["element_pruned_only"] += int(prune_stats.get("element_pruned_only", 0))
        self.stats["aggressive_blocks"] += int(prune_stats.get("aggressive_blocks", 0))
        self.stats["conservative_blocks"] += int(prune_stats.get("conservative_blocks", 0))
        self.stats["padded_elements"] += int(padded_elements)

    def summarize_stats(self) -> dict:
        if self.stats["total_rows"] == 0:
            return {}
        total_elements = self.stats["total_elements"]
        total_blocks = self.stats["total_blocks"]
        return {
            "calls": self.stats["calls"],
            "total_rows": self.stats["total_rows"],
            "total_elements": total_elements,
            "total_pruned_elements": self.stats["total_pruned_elements"],
            "total_pruned_rate": (
                self.stats["total_pruned_elements"] / total_elements if total_elements > 0 else 0.0
            ),
            "block_pruned_blocks": self.stats["block_pruned_blocks"],
            "total_blocks": total_blocks,
            "block_pruned_block_rate": (
                self.stats["block_pruned_blocks"] / total_blocks if total_blocks > 0 else 0.0
            ),
            "block_pruned_elements": self.stats["block_pruned_elements"],
            "block_pruned_element_rate": (
                self.stats["block_pruned_elements"] / total_elements if total_elements > 0 else 0.0
            ),
            "element_pruned_only": self.stats["element_pruned_only"],
            "element_pruned_only_rate": (
                self.stats["element_pruned_only"] / total_elements if total_elements > 0 else 0.0
            ),
            "aggressive_blocks": self.stats["aggressive_blocks"],
            "conservative_blocks": self.stats["conservative_blocks"],
            "padded_elements": self.stats["padded_elements"],
        }

    def __call__(self, input_tensor: torch.Tensor, dim: int = -1, _stacklevel: int = 3, dtype=None) -> torch.Tensor:
        normalized_dim = dim if dim >= 0 else input_tensor.dim() + dim
        if normalized_dim != input_tensor.dim() - 1:
            return self.fallback_softmax(input_tensor, dim=dim, _stacklevel=_stacklevel, dtype=dtype)

        target_dtype = dtype or input_tensor.dtype
        row_depth = int(input_tensor.shape[-1])
        if row_depth <= 0:
            return self.fallback_softmax(input_tensor, dim=dim, _stacklevel=_stacklevel, dtype=dtype)

        model = self._get_model(row_depth)
        block_size = model.cfg.block_size_effective
        pad_len = (block_size - (row_depth % block_size)) % block_size

        flat = input_tensor.detach().reshape(-1, row_depth)
        if input_tensor.device.type == "cpu" and input_tensor.dtype == torch.float32 and target_dtype == torch.float32:
            flat_np = flat.numpy()
        else:
            flat_np = flat.to(torch.float32).cpu().numpy()

        padded_elements = 0
        if pad_len > 0:
            flat_np = np.asarray(flat_np, dtype=np.float32)
            flat_np = np.pad(flat_np, ((0, 0), (0, pad_len)), mode="constant", constant_values=-float("inf"))
            padded_elements = flat_np.shape[0] * pad_len
            model = self._get_model(row_depth + pad_len)

        approx_rows, prune_stats = model.simulate_rows_fast(flat_np)
        self._aggregate_stats(prune_stats, padded_elements=padded_elements, row_count=int(flat_np.shape[0]))

        if pad_len > 0:
            approx_rows = approx_rows[:, :row_depth]

        if isinstance(approx_rows, torch.Tensor):
            approx_tensor = approx_rows.to(device=input_tensor.device, dtype=target_dtype)
        else:
            approx_tensor = torch.as_tensor(approx_rows, device=input_tensor.device, dtype=target_dtype)
        return approx_tensor.reshape_as(input_tensor)


@contextlib.contextmanager
def patched_softmax(profile: str | None, input_format: str = "fp16"):
    """Temporarily replace PyTorch softmax with the hardware approximation."""
    if profile in (None, "", "exact"):
        yield None
        return

    original_f_softmax = F.softmax
    original_torch_softmax = torch.softmax
    runtime = HardwareSoftmaxRuntime(
        profile=profile,
        input_format=input_format,
        fallback_softmax=original_f_softmax,
    )

    def replacement_f(input_tensor: torch.Tensor, dim: int = -1, _stacklevel: int = 3, dtype=None):
        return runtime(input_tensor, dim=dim, _stacklevel=_stacklevel, dtype=dtype)

    def replacement_torch(input_tensor: torch.Tensor, dim=None, dtype=None):
        dim = -1 if dim is None else dim
        return runtime(input_tensor, dim=dim, dtype=dtype)

    F.softmax = replacement_f
    torch.softmax = replacement_torch
    try:
        yield runtime
    finally:
        F.softmax = original_f_softmax
        torch.softmax = original_torch_softmax
