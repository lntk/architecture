import torch.nn as nn


def dense_block(in_dim, out_dim, activation="relu", normalize=True, dropout=True):
    layers = list()

    """ DENSE LAYER """
    layers.append(nn.Linear(in_dim, out_dim, bias=not normalize))

    """ BATCH NORMALIZATION LAYER """
    if normalize:
        layers.append(nn.BatchNorm1d(out_dim, 0.8))

    """ ACTIVATION LAYER """
    if activation is not None:
        layers.append(get_activation(name=activation))

    """ DROPOUT LAYER """
    if dropout:
        layers.append(nn.Dropout(0.2))

    return layers


def cnn_block(in_channels, out_channels, resizing="half", activation="relu", normalize=True, dropout=True):
    layers = list()

    """ CONVOLUTIONAL LAYER """
    if resizing == "half_3":
        layers.append(nn.Conv2d(in_channels, out_channels, 3, 2, 1, bias=not normalize))
    elif resizing == "half_4":
        layers.append(nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=not normalize))
    elif resizing == "same":
        layers.append(nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=not normalize))
    elif resizing == "double":
        layers.append(nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=not normalize))
    else:
        raise NotImplementedError

    """ BATCH NORMALIZATION LAYER """
    if normalize:
        layers.append(nn.BatchNorm2d(out_channels))

    """ ACTIVATION LAYER """
    if activation is not None:
        layers.append(get_activation(name=activation))

    """ DROPOUT LAYER """
    if dropout:
        layers.append(nn.Dropout(0.2))

    return layers


def get_activation(name):
    if name == "leaky_relu":
        activation_layer = nn.LeakyReLU(0.2, inplace=True)
    elif name == "relu":
        activation_layer = nn.ReLU(inplace=True)
    elif name == "elu":
        activation_layer = nn.ELU(inplace=True)
    elif name == "tanh":
        activation_layer = nn.Tanh()
    elif name == "sigmoid":
        activation_layer = nn.Sigmoid()
    elif name == "softplus":
        activation_layer = nn.Softplus(beta=100)
    else:
        raise NotImplementedError

    return activation_layer


def clamp_weights(net, clipping_value):
    for p in net.parameters():
        p.data.clamp_(-clipping_value, clipping_value)
