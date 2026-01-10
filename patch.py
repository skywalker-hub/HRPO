import types
import torch
from transformers.trainer import *
from transformers import TrainerCallback


class ThinkingModulesMonitorCallback(TrainerCallback):
    """
    Callback to monitor gradient norms of info_head and token_gate_matrix modules.
    
    Note: gate/g_k_mean, continuous_bias/norm are recorded directly in GRPOTrainer.
    This callback only handles gradient norms which need to be computed after backward.
    """
    
    def __init__(self, trainer_ref=None):
        self.trainer_ref = trainer_ref
        self._grad_stats = {
            'info_head/grad_norm': [],
            'gate/grad_norm': [],
        }
    
    def _compute_grad_norm(self, params):
        """Compute gradient L2 norm for a list of parameters."""
        total_norm = 0.0
        for p in params:
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        return total_norm ** 0.5
    
    def on_train_begin(self, args, state, control, model=None, **kwargs):
        """Save model reference at training start."""
        self._model = model
    
    def on_step_end(self, args, state, control, **kwargs):
        """Compute gradient norms after each training step."""
        model = self._model if hasattr(self, '_model') else kwargs.get('model')
        if model is None:
            return
        
        # Compute gradient norms for info_head and token_gate_matrix
        info_head_params = []
        token_gate_params = []
        
        for name, param in model.named_parameters():
            if 'info_head' in name and param.requires_grad:
                info_head_params.append(param)
            elif 'token_gate_matrix' in name and param.requires_grad:
                token_gate_params.append(param)
        
        if info_head_params:
            grad_norm = self._compute_grad_norm(info_head_params)
            self._grad_stats['info_head/grad_norm'].append(grad_norm)
        
        if token_gate_params:
            grad_norm = self._compute_grad_norm(token_gate_params)
            self._grad_stats['gate/grad_norm'].append(grad_norm)
    
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