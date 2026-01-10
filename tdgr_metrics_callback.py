import math
from typing import Optional

import torch
from transformers import TrainerCallback


class TDGRMetricsCallback(TrainerCallback):
    """
    Log Token-Dependent Gated Residual (TDGR) module metrics.

    Important: gradients are zeroed BEFORE Trainer.log() is called, so we must
    capture grad norms in on_pre_optimizer_step (after backward + grad clipping,
    before optimizer.step / zero_grad), then inject them into logs in on_log.
    """

    def __init__(self):
        super().__init__()
        self._last_info_head_grad_norm: Optional[float] = None
        self._last_gate_grad_norm: Optional[float] = None

    @staticmethod
    def _grad_l2_norm(model, name_substr: str) -> float:
        total_sq = 0.0
        for n, p in model.named_parameters():
            if name_substr not in n:
                continue
            if p.grad is None:
                continue
            g = p.grad.detach()
            # Be robust to mixed precision
            if g.is_floating_point():
                g = g.float()
            total_sq += g.pow(2).sum().item()
        return math.sqrt(total_sq) if total_sq > 0.0 else 0.0

    def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
        if model is None:
            return
        # Capture before optimizer.step() and before model.zero_grad().
        self._last_info_head_grad_norm = self._grad_l2_norm(model, "info_head")
        self._last_gate_grad_norm = self._grad_l2_norm(model, "token_gate_matrix")

    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        if not logs:
            return
        # Inject into the logs dict so it's printed and forwarded to W&B.
        if self._last_info_head_grad_norm is not None:
            logs["info_head/grad_norm"] = float(self._last_info_head_grad_norm)
        if self._last_gate_grad_norm is not None:
            logs["gate/grad_norm"] = float(self._last_gate_grad_norm)


def ensure_tdgr_metrics_callback(trainer) -> None:
    """
    Ensure TDGRMetricsCallback is registered BEFORE WandbCallback, so it can
    inject metrics into `logs` in time for W&B logging.
    """
    callbacks = getattr(trainer, "callback_handler", None)
    if callbacks is None:
        return

    cb_list = getattr(trainer.callback_handler, "callbacks", None)
    if cb_list is None:
        return

    for cb in cb_list:
        if isinstance(cb, TDGRMetricsCallback):
            return

    # Prepend so it runs early in on_log.
    cb_list.insert(0, TDGRMetricsCallback())

