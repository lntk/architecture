import torch.nn as nn
import torch
from architecture.util import cnn_block, dense_block


def get_encoder(net):
    if net == "mlp":
        return EncoderFc
    elif net == "gwgan":
        return EncoderGwGan
    elif net == "mmdgan":
        return EncoderMmdGan
    else:
        raise NotImplementedError


class EncoderVaeFc(nn.Module):
    def __init__(self, in_dim, out_dim, DIM=64):
        super(EncoderVaeFc, self).__init__()

        self.fc_shared = nn.Sequential(
            *dense_block(in_dim, DIM, activation="leaky_relu", normalize=True, dropout=False),
            *dense_block(DIM, DIM * 2, activation="leaky_relu", normalize=True, dropout=False),
            *dense_block(DIM * 2, DIM * 4, activation="leaky_relu", normalize=True, dropout=False),
        )

        self.fc_mean = nn.Sequential(
            nn.Linear(DIM * 4, out_dim),
        )

        self.fc_log_variance = nn.Sequential(
            nn.Linear(DIM * 4, out_dim),
        )

    @staticmethod
    def reparameterize(mean, log_variance):
        std = torch.exp(0.5 * log_variance)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x):
        x = self.fc_shared(x)

        mean, log_variance = self.fc_mean(x), self.fc_log_variance(x)

        z = self.reparameterize(mean, log_variance)

        return z, mean, log_variance


class EncoderFc(nn.Module):
    def __init__(self, in_dim, out_dim, DIM=64):
        super(EncoderFc, self).__init__()

        self.fc = nn.Sequential(
            *dense_block(in_dim, DIM, activation="leaky_relu", normalize=True, dropout=False),
            *dense_block(DIM, DIM * 2, activation="leaky_relu", normalize=True, dropout=False),
            *dense_block(DIM * 2, DIM * 4, activation="leaky_relu", normalize=True, dropout=False),
            *dense_block(DIM * 4, out_dim, activation=None, normalize=False, dropout=False),
        )

    def forward(self, x):        
        return self.fc(x)


class EncoderVaeCnn(nn.Module):
    def __init__(self, size_x, channel_x, dim_z, hidden_ae=16):
        super(EncoderVaeCnn, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(channel_x, hidden_ae, 4, 2, 1, bias=True),  # L x H/2 x W/2
            nn.LeakyReLU(0.2),
            nn.Conv2d(hidden_ae, hidden_ae * 2, 4, 2, 1, bias=False),  # 2L x H/4 x W/4
            nn.BatchNorm2d(hidden_ae * 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(hidden_ae * 2, hidden_ae * 4, 4, 2, 1, bias=False),  # 4L x H/8 x W/8
            nn.BatchNorm2d(hidden_ae * 4),
            nn.LeakyReLU(0.2),
            nn.Conv2d(hidden_ae * 4, hidden_ae * 8, 4, 2, 1, bias=False),  # 8L x H/16 x W/16
            nn.BatchNorm2d(hidden_ae * 8),
            nn.LeakyReLU(0.2),
        )

        self.fc_mean = nn.Sequential(
            nn.Linear((hidden_ae * 8) * (size_x // 16) * (size_x // 16), dim_z),
        )

        self.fc_log_variance = nn.Sequential(
            nn.Linear((hidden_ae * 8) * (size_x // 16) * (size_x // 16), dim_z),
        )

    @staticmethod
    def reparameterize(mean, log_variance):
        std = torch.exp(0.5 * log_variance)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.cnn(x)
        x = x.view(batch_size, -1)
        mean, log_variance = self.fc_mean(x), self.fc_log_variance(x)

        z = self.reparameterize(mean, log_variance)

        return z, mean, log_variance


class EncoderCnn(nn.Module):
    def __init__(self, size_x, channel_x, dim_z, hidden_ae=16):
        super(EncoderCnn, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(channel_x, hidden_ae, 4, 2, 1, bias=True),  # L x H/2 x W/2
            nn.LeakyReLU(0.2),
            nn.Conv2d(hidden_ae, hidden_ae * 2, 4, 2, 1, bias=False),  # 2L x H/4 x W/4
            nn.BatchNorm2d(hidden_ae * 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(hidden_ae * 2, hidden_ae * 4, 4, 2, 1, bias=False),  # 4L x H/8 x W/8
            nn.BatchNorm2d(hidden_ae * 4),
            nn.LeakyReLU(0.2),
            nn.Conv2d(hidden_ae * 4, hidden_ae * 8, 4, 2, 1, bias=False),  # 8L x H/16 x W/16
            nn.BatchNorm2d(hidden_ae * 8),
            nn.LeakyReLU(0.2),
        )

        self.fc = nn.Sequential(
            nn.Linear((hidden_ae * 8) * (size_x // 16) * (size_x // 16), 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, dim_z),
        )

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.cnn(x)
        x = x.view(batch_size, -1)

        z = self.fc(x)

        return z


# --- Architecture using in MMD-GAN: https://arxiv.org/abs/1705.08584 ---
# --- Source: https://github.com/OctoberChang/MMD-GAN ---
# --- Input: (batch_size, channel_x, size_x, size_x)
# --- Output: (batch_size, dim_z)
class EncoderMmdGan(nn.Module):
    def __init__(self, size_x, channel_x, dim_z=100, ndf=64):
        super(EncoderMmdGan, self).__init__()
        assert size_x % 16 == 0, "isize has to be a multiple of 16"

        # input is nc x isize x isize
        main = nn.Sequential()
        main.add_module('initial-conv-{0}-{1}'.format(channel_x, ndf),
                        nn.Conv2d(channel_x, ndf, 4, 2, 1, bias=False))
        main.add_module('initial-relu-{0}'.format(ndf),
                        nn.LeakyReLU(0.2, inplace=True))
        csize, cndf = size_x / 2, ndf

        while csize > 4:
            in_feat = cndf
            out_feat = cndf * 2
            main.add_module('pyramid-{0}-{1}-conv'.format(in_feat, out_feat),
                            nn.Conv2d(in_feat, out_feat, 4, 2, 1, bias=False))
            main.add_module('pyramid-{0}-batchnorm'.format(out_feat),
                            nn.BatchNorm2d(out_feat))
            main.add_module('pyramid-{0}-relu'.format(out_feat),
                            nn.LeakyReLU(0.2, inplace=True))
            cndf = cndf * 2
            csize = csize / 2

        main.add_module('final-{0}-{1}-conv'.format(cndf, 1),
                        nn.Conv2d(cndf, dim_z, 4, 1, 0, bias=False))

        self.main = main

    def forward(self, x):
        batch_size = x.shape[0]

        output = self.main(x)

        return output.view(batch_size, -1)


class EncoderMmdGan2(nn.Module):
    def __init__(self, size_x, channel_x, dim_z=100, ndf=64):
        super(EncoderMmdGan2, self).__init__()
        assert size_x % 16 == 0, "isize has to be a multiple of 16"

        # input is nc x isize x isize
        main = nn.Sequential()
        main.add_module('initial-conv-{0}-{1}'.format(channel_x, ndf),
                        nn.Conv2d(channel_x, ndf, 4, 2, 1, bias=False))
        main.add_module('initial-relu-{0}'.format(ndf),
                        nn.LeakyReLU(0.2, inplace=True))
        csize, cndf = size_x / 2, ndf

        self.cnn = nn.Sequential(
            nn.Conv2d(channel_x, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

        )

        while csize > 4:
            in_feat = cndf
            out_feat = cndf * 2
            main.add_module('pyramid-{0}-{1}-conv'.format(in_feat, out_feat),
                            nn.Conv2d(in_feat, out_feat, 4, 2, 1, bias=False))
            main.add_module('pyramid-{0}-batchnorm'.format(out_feat),
                            nn.BatchNorm2d(out_feat))
            main.add_module('pyramid-{0}-relu'.format(out_feat),
                            nn.LeakyReLU(0.2, inplace=True))
            cndf = cndf * 2
            csize = csize / 2

        main.add_module('final-{0}-{1}-conv'.format(cndf, 1),
                        nn.Conv2d(cndf, dim_z, 4, 1, 0, bias=False))

        self.main = main

    def forward(self, x):
        batch_size = x.shape[0]

        output = self.main(x)

        return output.view(batch_size, -1)


# class EncoderVAE(nn.Module):
#     def __init__(self, size_x, channel_x, dim_z):
#         super(EncoderVAE, self).__init__()
#
#         self.channel_x = channel_x
#
#         self.cnn = nn.Sequential(
#             *cnn_block(channel_x, 16, resizing="half_3", activation="leaky_relu", normalize=False, dropout=False),
#             *cnn_block(16, 32, resizing="half_3", activation="leaky_relu", normalize=True, dropout=False),
#             *cnn_block(32, 64, resizing="half_3", activation="leaky_relu", normalize=True, dropout=False),
#             *cnn_block(64, 128, resizing="half_3", activation="leaky_relu", normalize=True, dropout=False),
#         )
#
#         out_cnn_size = size_x // 16
#
#         self.fc_mean = nn.Sequential(
#             *dense_block(out_cnn_size * out_cnn_size * 128, dim_z, activation=None, normalize=False, dropout=False)
#         )
#
#         self.fc_log_variance = nn.Sequential(
#             *dense_block(out_cnn_size * out_cnn_size * 128, dim_z, activation=None, normalize=False, dropout=False)
#         )
#
#     @staticmethod
#     def reparameterize(mean, log_variance):
#         std = torch.exp(0.5 * log_variance)
#         eps = torch.randn_like(std)
#         return mean + eps * std
#
#     def forward(self, x):
#         batch_size = x.shape[0]
#         x = self.cnn(x)
#
#         x = x.view(batch_size, -1)
#         mean, log_variance = self.fc_mean(x), self.fc_log_variance(x)
#
#         z = self.reparameterize(mean, log_variance)
#
#         return z, mean, log_variance


class EncoderAE(nn.Module):
    def __init__(self, size_x, channel_x, dim_z):
        super(EncoderAE, self).__init__()

        self.channel_x = channel_x

        self.cnn = nn.Sequential(
            *cnn_block(channel_x, 32, resizing="half", activation="leaky_relu", normalize=True, dropout=True),
            *cnn_block(32, 32, resizing="half", activation="leaky_relu", normalize=True, dropout=True),
            *cnn_block(32, 64, resizing="half", activation="leaky_relu", normalize=True, dropout=True),
            *cnn_block(64, 64, resizing="half", activation="leaky_relu", normalize=True, dropout=True),
        )

        out_cnn_size = size_x // 16
        self.fc = nn.Sequential(
            *dense_block(out_cnn_size * out_cnn_size * 64, 256, activation="leaky_relu", normalize=True, dropout=True),
            nn.Linear(256, dim_z),
        )

    def forward(self, x):
        batch_size = x.shape[0]

        x = self.cnn(x)
        x = self.fc(x.view(batch_size, -1))

        return x


# --- Architecture of Gromov-Wasserstein GAN: https://arxiv.org/abs/1905.05461 ---
# --- Source: https://github.com/bunnech/gw_gan ---
class EncoderGwGan(nn.Module):
    def __init__(self, size_x=32, channel_x=1, dim_z=1024):
        super(EncoderGwGan, self).__init__()
        self.input_dim = channel_x
        self.output_dim = dim_z
        self.input_size = size_x

        self.conv = nn.Sequential(
            nn.Conv2d(self.input_dim, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * (self.input_size // 4) * (self.input_size // 4), 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, self.output_dim),
        )

    def forward(self, x):
        z = self.conv(x)
        z = z.view(-1, 128 * (self.input_size // 4) * (self.input_size // 4))
        z = self.fc(z)
        return z
