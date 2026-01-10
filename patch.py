import types
import torch
from transformers.trainer import *
from transformers import TrainerCallback


class ThinkingModulesMonitorCallback(TrainerCallback):
    """
    Callback to monitor gradient norms of info_head and token_gate_matrix modules.
    
    Uses backward hooks to capture gradients before they are cleared by zero_grad().
    """
    
    def __init__(self):
        self._grad_stats = {
            'info_head/grad_norm': [],
            'gate/grad_norm': [],
        }
        self._hooks = []
        self._current_info_head_grad_sq = 0.0
        self._current_gate_grad_sq = 0.0
    
    def _make_hook(self, module_type):
        """Create a hook function for the given module type."""
        def hook(grad):
            grad_norm_sq = grad.detach().norm(2).item() ** 2
            if module_type == 'info_head':
                self._current_info_head_grad_sq += grad_norm_sq
            else:
                self._current_gate_grad_sq += grad_norm_sq
        return hook
    
    def on_train_begin(self, args, state, control, model=None, **kwargs):
        """Register gradient hooks at training start."""
        if model is None:
            return
        
        # Register hooks on parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                if 'info_head' in name:
                    hook = param.register_hook(self._make_hook('info_head'))
                    self._hooks.append(hook)
                elif 'token_gate_matrix' in name:
                    hook = param.register_hook(self._make_hook('gate'))
                    self._hooks.append(hook)
    
    def on_step_end(self, args, state, control, **kwargs):
        """Collect gradient norms after backward (hooks already captured them)."""
        # Record the accumulated gradient norms
        if self._current_info_head_grad_sq > 0:
            self._grad_stats['info_head/grad_norm'].append(self._current_info_head_grad_sq ** 0.5)
        if self._current_gate_grad_sq > 0:
            self._grad_stats['gate/grad_norm'].append(self._current_gate_grad_sq ** 0.5)
        
        # Reset for next step
        self._current_info_head_grad_sq = 0.0
        self._current_gate_grad_sq = 0.0
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Add gradient norms to logs when logging occurs."""
        if logs is None:
            return
        
        # Average and add to logs
        for key, values in self._grad_stats.items():
            if values:
                logs[key] = sum(values) / len(values)
        
        # Clear after logging
        for key in self._grad_stats:
            self._grad_stats[key] = []
    
    def on_train_end(self, args, state, control, **kwargs):
        """Remove hooks at training end."""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []


def patch_trainer_optimizer(trainer, lr_info_head=1e-4, lr_token_gate_matrix=1e-4):
    """
    Patch the trainer optimizer for Token-Dependent Gated Residual mechanism.
    
    Args:
        trainer: The trainer instance
        lr_info_head: Learning rate for info_head linear layer (default: 1e-4)
        lr_token_gate_matrix: Learning rate for token_gate_matrix embedding (default: 1e-4)
    """
    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model

        if self.optimizer is None:
            decay_parameters = self.get_decay_parameter_names(opt_model)
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if ("info_head" not in n and "token_gate_matrix" not in n and n in decay_parameters and p.requires_grad)
                    ],
                    "lr": self.args.learning_rate,
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if ("info_head" not in n and "token_gate_matrix" not in n and n not in decay_parameters and p.requires_grad)
                    ],
                    "lr": self.args.learning_rate,
                    "weight_decay": 0.0,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if ("info_head" in n and p.requires_grad)
                    ],
                    "lr": lr_info_head,
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if ("token_gate_matrix" in n and p.requires_grad)
                    ],
                    "lr": lr_token_gate_matrix,
                    "weight_decay": self.args.weight_decay,
                },
            ]

            if self.optimizer_cls_and_kwargs is not None:
                optimizer_cls, optimizer_kwargs = self.optimizer_cls_and_kwargs
            else:
                optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args, opt_model)

            # Overwrite `params` in case it's created by `get_optimizer_cls_and_kwargs`
            # e.g. for GaLore optimizer.
            if "params" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("params")

            # Overwrite `model` in case it's created by `get_optimizer_cls_and_kwargs`
            # e.g. for LOMO optimizer.
            if "model" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("model")

            # For layer-wise dummy optimizers we overwrite optimizer_grouped_parameters with `optimizer_dict`
            # to avoid arguments conflicts.
            if "optimizer_dict" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("optimizer_dict")

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped / 2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped / 2**20}M params")

        if is_sagemaker_mp_enabled():
            self.optimizer = smp.DistributedOptimizer(self.optimizer)

        return self.optimizer

    trainer._old_create_optimizer = trainer.create_optimizer
    trainer.create_optimizer = types.MethodType(create_optimizer, trainer)