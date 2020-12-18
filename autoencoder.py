import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(Autoencoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        batch_size = x.shape[0]

        z = self.encoder(x)
        y = self.decoder(z)

        z = z.view(batch_size, -1)  # batch_size x dim_z
        y = y.view(batch_size, -1)  # batch_size x dim_y
        return z, y
