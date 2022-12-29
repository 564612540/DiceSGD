import torch
from typing import Callable, List, Optional, Union
from torch.optim import Optimizer
from opacus.optimizers.optimizer import DPOptimizer
from opt_einsum.contract import contract

def _mark_as_processed(obj: Union[torch.Tensor, List[torch.Tensor]]):
    """
    Marks parameters that have already been used in the optimizer step.
    DP-SGD puts certain restrictions on how gradients can be accumulated. In particular,
    no gradient can be used twice - client must call .zero_grad() between
    optimizer steps, otherwise privacy guarantees are compromised.
    This method marks tensors that have already been used in optimizer steps to then
    check if zero_grad has been duly called.
    Notes:
          This is used to only mark ``p.grad_sample`` and ``p.summed_grad``
    Args:
        obj: tensor or a list of tensors to be marked
    """

    if isinstance(obj, torch.Tensor):
        obj._processed = True
    elif isinstance(obj, list):
        for x in obj:
            x._processed = True

def _check_processed_flag_tensor(x: torch.Tensor):
    """
    Checks if this gradient tensor has been previously used in optimization step.
    See Also:
        :meth:`~opacus.optimizers.optimizer._mark_as_processed`
    Args:
        x: gradient tensor
    Raises:
        ValueError
            If tensor has attribute ``._processed`` previously set by
            ``_mark_as_processed`` method
    """

    if hasattr(x, "_processed"):
        raise ValueError(
            "Gradients haven't been cleared since the last optimizer step. "
            "In order to obtain privacy guarantees you must call optimizer.zero_grad()"
            "on each step"
        )

def _check_processed_flag(obj: Union[torch.Tensor, List[torch.Tensor]]):
    """
    Checks if this gradient tensor (or a list of tensors) has been previously
    used in optimization step.
    See Also:
        :meth:`~opacus.optimizers.optimizer._mark_as_processed`
    Args:
        x: gradient tensor or a list of tensors
    Raises:
        ValueError
            If tensor (or at least one tensor from the list) has attribute
            ``._processed`` previously set by ``_mark_as_processed`` method
    """

    if isinstance(obj, torch.Tensor):
        _check_processed_flag_tensor(obj)
    elif isinstance(obj, list):
        for x in obj:
            _check_processed_flag_tensor(x)

def _generate_noise(
    std: float,
    reference: torch.Tensor,
    generator=None,
    secure_mode: bool = False,
) -> torch.Tensor:
    """
    Generates noise according to a Gaussian distribution with mean 0
    Args:
        std: Standard deviation of the noise
        reference: The reference Tensor to get the appropriate shape and device
            for generating the noise
        generator: The PyTorch noise generator
        secure_mode: boolean showing if "secure" noise need to be generated
            (see the notes)
    Notes:
        If `secure_mode` is enabled, the generated noise is also secure
        against the floating point representation attacks, such as the ones
        in https://arxiv.org/abs/2107.10138 and https://arxiv.org/abs/2112.05307.
        The attack for Opacus first appeared in https://arxiv.org/abs/2112.05307.
        The implemented fix is based on https://arxiv.org/abs/2107.10138 and is
        achieved through calling the Gaussian noise function 2*n times, when n=2
        (see section 5.1 in https://arxiv.org/abs/2107.10138).
        Reason for choosing n=2: n can be any number > 1. The bigger, the more
        computation needs to be done (`2n` Gaussian samples will be generated).
        The reason we chose `n=2` is that, `n=1` could be easy to break and `n>2`
        is not really necessary. The complexity of the attack is `2^p(2n-1)`.
        In PyTorch, `p=53` and so complexity is `2^53(2n-1)`. With `n=1`, we get
        `2^53` (easy to break) but with `n=2`, we get `2^159`, which is hard
        enough for an attacker to break.
    """
    zeros = torch.zeros(reference.shape, device=reference.device)
    if std == 0:
        return zeros
    # TODO: handle device transfers: generator and reference tensor
    # could be on different devices
    if secure_mode:
        torch.normal(
            mean=0,
            std=std,
            size=(1, 1),
            device=reference.device,
            generator=generator,
        )  # generate, but throw away first generated Gaussian sample
        sum = zeros
        for _ in range(4):
            sum += torch.normal(
                mean=0,
                std=std,
                size=reference.shape,
                device=reference.device,
                generator=generator,
            )
        return sum / 2
    else:
        return torch.normal(
            mean=0,
            std=std,
            size=reference.shape,
            device=reference.device,
            generator=generator,
        )

class DiceSGD(DPOptimizer):
    def __init__(self, optimizer: Optimizer, *, noise_multiplier: float, max_grad_norm: float, expected_batch_size: Optional[int], loss_reduction: str = "mean", generator=None, secure_mode: bool = False):
        super().__init__(optimizer, noise_multiplier=noise_multiplier, max_grad_norm=max_grad_norm, expected_batch_size=expected_batch_size, loss_reduction = loss_reduction, generator = generator, secure_mode = secure_mode)
        for p in self.params:
            p.feedback_error = None
            p.past_error = None

    def clip_and_accumulate(self):
        """
        Performs gradient clipping.
        Stores clipped and aggregated gradients into `p.summed_grad```
        Stores clipping error into `p.feedback_error`
        """
        if len(self.grad_samples[0]) == 0:
            # Empty batch
            per_sample_clip_factor = torch.zeros((0,))
        else:
            per_param_norms = [
                g.reshape(len(g), -1).norm(2, dim=-1) for g in self.grad_samples
            ]
            per_sample_norms = torch.stack(per_param_norms, dim=1).norm(2, dim=1)
            per_sample_clip_factor = (
                self.max_grad_norm / (per_sample_norms + 1e-6)
            ).clamp(max=1.0)

        for p in self.params:
            _check_processed_flag(p.grad_sample)
            grad_sample = self._get_flat_grad_sample(p)
            grad = contract("i,i...", per_sample_clip_factor, grad_sample)
            sum_grad = contract("i...->...", grad_sample)

            if p.past_error is None:
                if p.feedback_error is not None:
                    p.past_eror = p.feedback_error
            if p.summed_grad is not None:
                p.summed_grad += grad
                p.feedback_error += (sum_grad-grad)
            else:
                p.summed_grad = grad
                p.feedback_error = (sum_grad - grad)

            _mark_as_processed(p.grad_sample)

    def scale_grad(self):
        """
        Applies given ``loss_reduction`` to ``p.grad``.
        Additionally add the feedback error to the gradients
        """
        if self.params[0].past_error is not None:
            per_param_norm = [p.past_error.reshape(len(p.past_error), -1).norm(2, dim=-1) for p in self.params]
            feedback_error_norm = torch.stack(per_param_norm, dim=1).norm(2, dim=1)
            clip_factor = (self.max_grad_norm / (feedback_error_norm + 1e-6)).clamp(max=1.0)
        if self.loss_reduction == "mean":
            for p in self.params:
                p.grad /= self.expected_batch_size * self.accumulated_iterations
                if p.past_error is not None:
                    feedback_error = p.past_error*clip_factor
                    noise = _generate_noise(
                        std=self.noise_multiplier * self.max_grad_norm,
                        reference=p.summed_grad,
                        generator=self.generator,
                        secure_mode=self.secure_mode,
                    )
                    p.grad += (feedback_error + noise).view_as(p)
                    p.feedback_error += p.past_error - feedback_error
        else:
            for p in self.params:
                if p.past_error is not None:
                    feedback_error = p.past_error*clip_factor
                    noise = _generate_noise(
                        std=self.noise_multiplier * self.max_grad_norm,
                        reference=p.summed_grad,
                        generator=self.generator,
                        secure_mode=self.secure_mode,
                    )
                    p.grad += (feedback_error+noise) * self.expected_batch_size * self.accumulated_iterations
                    p.feedback_error += p.past_error - feedback_error

    def zero_grad(self, set_to_none: bool = False):
        for p in self.params:
            p.past_error = None
        return super().zero_grad(set_to_none)