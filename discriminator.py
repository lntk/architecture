import torch.nn as nn
from architecture.util import cnn_block, dense_block


class DiscriminatorWGANGP(nn.Module):
    def __init__(self, args, DIM=16):
        super(DiscriminatorWGANGP, self).__init__()

        modules = [
            nn.Conv2d(args.channel_y, DIM, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(DIM, DIM * 2, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(DIM * 2, DIM * 4, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        if args.size_y == 32:
            modules += [
                nn.Conv2d(DIM * 4, 1, 4, 1, 0),
            ]
        elif args.size_y == 64:
            modules += [
                nn.Conv2d(DIM * 4, DIM * 8, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(DIM * 8, 1, 4, 1, 0),
            ]
        else:
            raise NotImplementedError

        self.cnn = nn.Sequential(*modules)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.cnn(x)

        return x.view(batch_size, 1)


class DiscriminatorFc(nn.Module):
    def __init__(self, args, DIM=64):
        super(DiscriminatorFc, self).__init__()

        self.fc = nn.Sequential(
            *dense_block(args.dim_y, DIM, activation="leaky_relu", normalize=False, dropout=False),
            *dense_block(DIM, 1, activation=None, normalize=False, dropout=False)
        )

    def forward(self, x):
        return self.fc(x)


# --- Architecture of DCGAN: https://arxiv.org/abs/1511.06434 ---
# --- Source: https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/dcgan/dcgan.py#L73 ---
class DiscriminatorDcGan(nn.Module):
    def __init__(self, size_x, channel_x):
        super(DiscriminatorDcGan, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(channel_x, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = size_x // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity
