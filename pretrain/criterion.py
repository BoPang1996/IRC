import torch
from torch import nn


class NCELoss(nn.Module):
    """
    Compute the PointInfoNCE loss
    """

    def __init__(self, temperature):
        super(NCELoss, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()
        self._target = {}

    def get_target(self, A):
        N, M = A.size()
        key = '%s:%s' % (str(A.device), N)

        if key not in self._target:
            self._target[key] = torch.arange(A.size(0), device=A.device).long()  # H*W

        return self._target[key]

    def forward(self, k, q):
        logits = torch.mm(k, q.transpose(1, 0))
        target = self.get_target(k)
        out = torch.div(logits, self.temperature)
        out = out.contiguous()

        loss = self.criterion(out, target)
        return loss
