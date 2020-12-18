import torch.nn as nn
from architecture.util import cnn_block, dense_block


def get_decoder(net):
    if net == "mlp":
        return DecoderFc
    elif net == "gwgan":
        return DecoderGwGan
    elif net == "mmdgan":
        return DecoderMmdGan
    elif net == "dcgan":
        return DecoderDcGan
    else:
        raise NotImplementedError


# class Decoder(nn.Module):
#     def __init__(self, size_y, channel_y, dim_z, hidden_ae=16):
#         super(Decoder, self).__init__()
#
#         self.size_y = size_y
#         self.dim_h = hidden_ae
#
#         self.fc = nn.Sequential(
#             nn.Linear(dim_z, (hidden_ae * 8) * (self.size_y // 8) * (self.size_y // 8)),
#             nn.ReLU()
#         )
#
#         self.cnn = nn.Sequential(
#             nn.ConvTranspose2d(hidden_ae * 8, hidden_ae * 4, 4, 2, 1, bias=False),  # 4L x H/4 x W/4
#             nn.BatchNorm2d(hidden_ae * 4),
#             nn.LeakyReLU(0.2),
#             nn.ConvTranspose2d(hidden_ae * 4, hidden_ae * 2, 4, 2, 1, bias=False),  # 2L x H/2 x W/2
#             nn.BatchNorm2d(hidden_ae * 2),
#             nn.LeakyReLU(0.2),
#             nn.ConvTranspose2d(hidden_ae * 2, channel_y, 4, 2, 1),  # C x H x W
#             nn.Tanh()
#         )
#
#     def forward(self, x):
#         x = self.fc(x)
#         x = x.view(-1, self.dim_h * 8, (self.size_y // 8), (self.size_y // 8))
#         x = self.cnn(x)
#         return x


class DecoderCnn(nn.Module):
    def __init__(self, size_y, channel_y, dim_z, final_activation="sigmoid"):
        super(DecoderCnn, self).__init__()

        out_cnn_size = size_y // 8
        self.out_cnn_size = out_cnn_size

        self.fc = nn.Sequential(
            *dense_block(dim_z, out_cnn_size * out_cnn_size * 128, activation="relu", normalize=True, dropout=False),
        )

        self.cnn = nn.Sequential(
            *cnn_block(128, 64, resizing="double", activation="leaky_relu", normalize=True, dropout=False),
            *cnn_block(64, 32, resizing="double", activation="leaky_relu", normalize=True, dropout=False),
            *cnn_block(32, channel_y, resizing="double", activation=final_activation, normalize=False, dropout=False),
        )

    def forward(self, x):
        batch_size = x.shape[0]

        x = self.fc(x)
        x = x.view(batch_size, 128, self.out_cnn_size, self.out_cnn_size)
        y = self.cnn(x)

        return y


class DecoderFc(nn.Module):
    def __init__(self, in_dim, out_dim, final_activation="tanh", DIM=64):
        super(DecoderFc, self).__init__()

        self.fc = nn.Sequential(
            *dense_block(in_dim, DIM, activation="leaky_relu", normalize=True, dropout=False),
            *dense_block(DIM, DIM * 2, activation="leaky_relu", normalize=True, dropout=False),
            *dense_block(DIM * 2, out_dim, activation=final_activation, normalize=False, dropout=False),
        )

    def forward(self, x):
        return self.fc(x)


# --- Architecture using in MMD-GAN: https://arxiv.org/abs/1705.08584 ---
# --- Source: https://github.com/OctoberChang/MMD-GAN ---
# --- Input: (batch_size, dim_z)
# --- Output: (batch_size, channel_y, size_y, size_y)
class DecoderMmdGan(nn.Module):
    def __init__(self, size_y, channel_y, dim_z=100, ngf=64):
        super(DecoderMmdGan, self).__init__()
        assert size_y % 16 == 0, "isize has to be a multiple of 16"

        cngf, tisize = ngf // 2, 4
        while tisize != size_y:
            cngf = cngf * 2
            tisize = tisize * 2

        main = nn.Sequential()
        main.add_module('initial-{0}-{1}-convt'.format(dim_z, cngf), nn.ConvTranspose2d(dim_z, cngf, 4, 1, 0, bias=False))
        main.add_module('initial-{0}-batchnorm'.format(cngf), nn.BatchNorm2d(cngf))
        main.add_module('initial-{0}-relu'.format(cngf), nn.ReLU(True))

        csize = 4
        while csize < size_y // 2:
            main.add_module('pyramid-{0}-{1}-convt'.format(cngf, cngf // 2),
                            nn.ConvTranspose2d(cngf, cngf // 2, 4, 2, 1, bias=False))
            main.add_module('pyramid-{0}-batchnorm'.format(cngf // 2),
                            nn.BatchNorm2d(cngf // 2))
            main.add_module('pyramid-{0}-relu'.format(cngf // 2),
                            nn.ReLU(True))
            cngf = cngf // 2
            csize = csize * 2

        main.add_module('final-{0}-{1}-convt'.format(cngf, channel_y), nn.ConvTranspose2d(cngf, channel_y, 4, 2, 1, bias=False))
        main.add_module('final-{0}-tanh'.format(channel_y),
                        nn.Tanh())

        self.main = main

    def forward(self, z):
        batch_size, dim_z = z.shape[:2]

        output = self.main(z.view(batch_size, dim_z, 1, 1))

        return output


# --- Architecture of DCGAN: https://arxiv.org/abs/1511.06434 ---
# --- Source: https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/dcgan/dcgan.py#L45-L70 ---
class DecoderDcGan(nn.Module):
    def __init__(self, size_y, channel_y, dim_z):
        super(DecoderDcGan, self).__init__()

        self.init_size = size_y // 4
        self.l1 = nn.Sequential(nn.Linear(dim_z, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, channel_y, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img  


# --- Architecture of Gromov-Wasserstein GAN: https://arxiv.org/abs/1905.05461 ---
# --- Source: https://github.com/bunnech/gw_gan ---
class DecoderGwGan(nn.Module):
    def __init__(self, size_y=32, channel_y=1, dim_z=100):
        super(DecoderGwGan, self).__init__()
        self.input_dim = dim_z
        self.output_dim = channel_y
        self.input_size = size_y

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 128 * (self.input_size // 4) * (self.input_size // 4)),
            nn.BatchNorm1d(128 * (self.input_size // 4) * (self.input_size // 4)),
            nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, self.output_dim, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 128, (self.input_size // 4), (self.input_size // 4))
        x = self.deconv(x)
        return x
