import torch
import torch.distributed as dist


class GatherLayer(torch.autograd.Function):
    """Gather tensors from all process, supporting backward propagation."""

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)

        # get sizes
        B = torch.tensor(input.size(0), dtype=torch.int64, device=input.device)
        all_B = [torch.zeros(1, dtype=torch.int64, device=input.device) for _ in range(dist.get_world_size())]
        dist.all_gather(all_B, B)
        max_B = max(all_B)

        # pad
        pad = torch.zeros((int(max_B - B), input.size(1)), device=input.device)
        input = torch.cat((input, pad), dim=0)

        # gather
        output = [torch.zeros((int(max_B), input.size(1)), device=input.device) for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)

        # remove pad and cat
        output = [out[:all_B[i]] for i, out in enumerate(output)]
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        (input,) = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out