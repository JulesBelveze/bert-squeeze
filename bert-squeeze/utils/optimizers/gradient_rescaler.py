import torch
from typing import Any
from overrides import overrides


class GradientRescaleFunction(torch.autograd.Function):

    @staticmethod
    @overrides
    def forward(ctx, inputs, weight, *args: Any, **kwargs: Any):
        ctx.save_for_backward(inputs)
        ctx.gd_scale_weight = weight
        output = inputs
        return output

    @staticmethod
    @overrides
    def backward(ctx, grad_output, *args: Any, **kwargs: Any):
        inputs = ctx.saved_tensors
        grad_input = grad_weight = None

        if ctx.needs_input_grad[0]:
            grad_input = ctx.gd_scale_weight * grad_output

        return grad_input, grad_weight


gradient_rescale = GradientRescaleFunction.apply
