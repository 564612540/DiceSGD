from typing import Optional, Sequence, Union
from fastDP.accounting import accounting_manager
import types
import torch
import math
from typing import Dict, Optional, Sequence, Union
from torch import nn
from torch.optim import Optimizer
from fastDP import PrivacyEngine
from opacus.accountants.utils import get_noise_multiplier
from fastDP import autograd_grad_sample

class PrivacyEngine_Dice(PrivacyEngine):
    def __init__(self, *args, error_max_grad_norm = 1.0, **kwargs):
        super(PrivacyEngine_Dice, self).__init__(*args, **kwargs)
        self.first_minibatch = True
        self.error_max_grad_norm = error_max_grad_norm
    def attach_dice(self, optimizer):
        autograd_grad_sample.add_hooks(model=self.module, loss_reduction=self.loss_reduction, 
                                       clipping_mode=self.clipping_mode, bias_only=self.bias_only,
                                       clipping_style=self.clipping_style, block_heads=self.block_heads,
                                       named_params=self.named_params, named_layers=self.named_layers,
                                       clipping_fn=self.clipping_fn, 
                                       numerical_stability_constant=self.numerical_stability_constant,
                                       max_grad_norm_layerwise=self.max_grad_norm_layerwise)

        # Override step.
        def dp_step(_self, **kwargs):
            closure = kwargs.pop("closure", None)
            
            # _self.zero_grad()         # make sure no non-private grad remains
            _self.privacy_engine._create_noisy_clipped_dice_gradient(**kwargs)
            _self.original_step(closure=closure)
            _self.privacy_engine.unlock()  # Only enable creating new grads once parameters are updated.
            _self.privacy_engine.steps += 1

        def get_privacy_spent(_self, **kwargs):
            return _self.privacy_engine.get_privacy_spent(**kwargs)

        def get_training_stats(_self, **kwargs):
            return _self.privacy_engine.get_training_stats(**kwargs)

        optimizer.privacy_engine = self

        optimizer.original_step = optimizer.step
        optimizer.step = types.MethodType(dp_step, optimizer)

        # Make getting info easier.
        optimizer.get_privacy_spent = types.MethodType(get_privacy_spent, optimizer)
        optimizer.get_training_stats = types.MethodType(get_training_stats, optimizer)

        self.module.privacy_engine = self

        # For easy detaching.
        self.optimizer = optimizer

    def _create_noisy_clipped_dice_gradient(self):
        """Create noisy clipped gradient for `optimizer.step`."""
        
        unsupported_param_name=[]
        for name,param in list(self.named_params):#https://thispointer.com/python-remove-elements-from-a-list-while-iterating/#1
            if not hasattr(param, 'summed_clipped_grad'):
                unsupported_param_name.append(name)
                self.named_params.remove((name,param)) # very helpful for models that are not 100% supported, e.g. in timm
            elif param.grad is None:
                print(name," this parameter has summed clipped grad but does not have grad")
        if unsupported_param_name!=[]:
            print(unsupported_param_name, 'are not supported by privacy engine; these parameters are not requiring gradient nor updated.')
                
        signals, noises, error_norms, grad_norms = [], [], [], []

        for name,param in self.named_params:
            if param.requires_grad:
                if hasattr(param,'error'):
                    first_minibatch = False
                    error_norms.append(param.error.reshape(-1).norm(2))
                else:
                    # param.error = None
                    first_minibatch = True
                    error_norms.append(torch.tensor(0.))
                # grad_norms.append(param.grad.reshape(-1).norm(2))
        error_norm = torch.stack(error_norms).norm(2) + 1e-6
        # grad_norm = torch.stack(grad_norms).norm(2)
        # print(error_norm.item(), grad_norm.item())
        
        for name,param in self.named_params:
            grad_diff = (param.grad-param.summed_clipped_grad)
            param.grad = param.summed_clipped_grad  # Ultra important to override `.grad`.
            del param.summed_clipped_grad

            if first_minibatch:
                param.error=grad_diff
            else:
                param.grad += param.error*torch.clamp_max(self.max_grad_norm/error_norm*self.error_max_grad_norm,1)
                param.error=param.error*torch.clamp_max(self.max_grad_norm/error_norm*self.error_max_grad_norm,(1-torch.clamp_max(self.max_grad_norm/error_norm*self.error_max_grad_norm,1)))+grad_diff
                # param.error=grad_diff
            del grad_diff

            if self.record_snr:
                signals.append(param.grad.reshape(-1).norm(2))

            if self.noise_multiplier > 0 and self.max_grad_norm > 0:
                noise = torch.normal(
                    mean=0,
                    std=self.noise_multiplier * self.max_grad_norm * math.sqrt(1+2*self.error_max_grad_norm),
                    size=param.size(),
                    device=param.device,
                    dtype=param.dtype,
                )
                param.grad += noise
                if self.record_snr:
                    noises.append(noise.reshape(-1).norm(2))
                del noise
            if self.loss_reduction=='mean':
                param.grad /= self.batch_size                

        if self.record_snr and len(noises) > 0:
            self.signal, self.noise = tuple(torch.stack(lst).norm(2).item() for lst in (signals, noises))
            self.noise_limit = math.sqrt(self.num_params) * self.noise_multiplier * self.max_grad_norm
            self.snr = self.signal / self.noise
        else:
            self.snr = math.inf  # Undefined!

        self.lock() 