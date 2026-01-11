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
        # Fixed sampling indices for cheap "mean trend" logging
        self._info_head_sample_idx: Optional[torch.Tensor] = None
        self._token_gate_flat_idx: Optional[torch.Tensor] = None
        self._token_gate_row_idx: Optional[torch.Tensor] = None

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

    @staticmethod
    def _unwrap_model(model):
        # Handle DDP/accelerate wrappers
        if hasattr(model, "module"):
            return model.module
        return model

    @staticmethod
    def _find_param_best(model, name_substr: str, prefer_substrings: Optional[list[str]] = None):
        """
        Return (name, param) for a parameter whose name contains name_substr and ends with 'weight'.
        If prefer_substrings is provided, pick the first match that contains any preferred substring (in order),
        otherwise fall back to the first match.
        """
        candidates = []
        for n, p in model.named_parameters():
            if name_substr in n and n.endswith("weight"):
                candidates.append((n, p))
        if not candidates:
            return None, None
        if prefer_substrings:
            for pref in prefer_substrings:
                for n, p in candidates:
                    if pref in n:
                        return n, p
        return candidates[0]
        return None, None

    def _sampled_mean(self, w: torch.Tensor, max_elems: int, cached_idx_attr: str) -> float:
        """
        Cheap mean over a fixed subset of elements (trend-friendly).
        """
        with torch.no_grad():
            w_flat = w.detach()
            if w_flat.is_floating_point():
                w_flat = w_flat.float()
            w_flat = w_flat.reshape(-1)
            n = w_flat.numel()
            if n == 0:
                return 0.0
            k = min(int(max_elems), int(n))
            idx = getattr(self, cached_idx_attr)
            if idx is None or idx.numel() != k or idx.device != w_flat.device:
                idx = torch.randint(0, n, (k,), device=w_flat.device)
                setattr(self, cached_idx_attr, idx)
            return w_flat.index_select(0, idx).mean().item()

    def _sampled_sigmoid_mean_token_gate(self, w: torch.Tensor, max_rows: int) -> float:
        """
        Cheap mean(sigmoid(token_gate_matrix)) over fixed sampled vocab rows.
        This avoids reducing over the full vocab_size*hidden_size every step.
        """
        with torch.no_grad():
            ww = w.detach()
            if ww.is_floating_point():
                ww = ww.float()
            if ww.dim() != 2:
                # Unexpected shape
                return torch.sigmoid(ww).mean().item()
            vocab = ww.shape[0]
            if vocab == 0:
                return 0.0
            k = min(int(max_rows), int(vocab))
            idx = self._token_gate_row_idx
            if idx is None or idx.numel() != k or idx.device != ww.device:
                idx = torch.randint(0, vocab, (k,), device=ww.device)
                self._token_gate_row_idx = idx
            sample = ww.index_select(0, idx)  # (k, hidden)
            return torch.sigmoid(sample).mean().item()

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

        # "Simple means" requested: track mean trends of the 2 matrices.
        if model is None:
            return
        m = self._unwrap_model(model)

        # info_head weight mean (sampled)
        _, info_w = self._find_param_best(
            m,
            "info_head",
            prefer_substrings=["modules_to_save.default.weight", "modules_to_save.default"],
        )
        if info_w is not None:
            logs["info_head/weight_mean"] = float(self._sampled_mean(info_w, max_elems=200_000, cached_idx_attr="_info_head_sample_idx"))

        # token_gate_matrix weight mean (sampled) + gate/g_k_mean as sigmoid(weight) mean (sampled rows)
        _, gate_w = self._find_param_best(
            m,
            "token_gate_matrix",
            prefer_substrings=["modules_to_save.default.weight", "modules_to_save.default"],
        )
        if gate_w is not None:
            logs["token_gate_matrix/weight_mean"] = float(
                self._sampled_mean(gate_w, max_elems=200_000, cached_idx_attr="_token_gate_flat_idx")
            )
            # Override the previous gate/g_k_mean (which was trying to be per-thinking-token) with a simple global trend.
            logs["gate/g_k_mean"] = float(self._sampled_sigmoid_mean_token_gate(gate_w, max_rows=2048))
            # Provide a stable proxy for continuous_bias norm trend (avoids dependence on thinking_embeds plumbing).
            # If v_t_norm has RMS=1, then ||(1/H)*v_t_norm*g_k|| ≈ (1/H)*sqrt(sum(g_k^2)).
            try:
                with torch.no_grad():
                    ww = gate_w.detach()
                    if ww.is_floating_point():
                        ww = ww.float()
                    if ww.dim() == 2 and ww.shape[0] > 0:
                        vocab = ww.shape[0]
                        k = min(2048, vocab)
                        idx = self._token_gate_row_idx
                        if idx is None or idx.numel() != k or idx.device != ww.device:
                            idx = torch.randint(0, vocab, (k,), device=ww.device)
                            self._token_gate_row_idx = idx
                        sample = ww.index_select(0, idx)
                        gk = torch.sigmoid(sample)
                        hd = sample.shape[-1]
                        proxy = (gk.pow(2).sum(dim=-1).sqrt() / float(hd)).mean().item()
                        logs["continuous_bias/norm"] = float(proxy)
                        logs["continuous_bias/norm_is_proxy"] = 1.0
            except Exception:
                pass

        # tdgr_alpha value (if present)
        try:
            # Prefer modules_to_save copy if PEFT wrapped.
            alpha_mod = None
            for name, mod in m.named_modules():
                if name.endswith("tdgr_alpha.modules_to_save.default"):
                    alpha_mod = mod
                    break
            if alpha_mod is None and hasattr(m, "tdgr_alpha"):
                alpha_mod = getattr(m, "tdgr_alpha")
                if hasattr(alpha_mod, "modules_to_save"):
                    alpha_mod = alpha_mod.modules_to_save.default
            if alpha_mod is not None and hasattr(alpha_mod, "log_alpha"):
                with torch.no_grad():
                    logs["tdgr_alpha/value"] = float(alpha_mod.log_alpha.detach().float().exp().item())
        except Exception:
            pass


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

