import torch.nn as nn
import torch
import torch.nn.functional as F


class WeightNet(nn.Module):
    def __init__(self, size_x, channel_x):
        super(WeightNet, self).__init__()
        ndf = 128
        nc = channel_x

        out_dim = 1

        self.network = nn.Sequential(
            nn.Conv2d(nc, ndf, 3, 1, 1, bias=True),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(ndf, 2 * ndf, 3, 1, 1, bias=True),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(2 * ndf, 4 * ndf, 3, 1, 1, bias=True),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(4 * ndf, out_dim, 4, 1, 0, bias=True)
        )

    def forward(self, x, label=None):
        weight_logits = self.network(x)
        weight_logits = torch.sum(weight_logits, (2, 3))
        weight_logits = weight_logits.view(weight_logits.size(0), -1)

        return F.relu(weight_logits)
