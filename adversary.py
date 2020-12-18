import torch.nn as nn
from architecture.util import cnn_block, dense_block


def adversary_function(adversary_type):
    adversary_functions = {
        "none": None,
        "same": SameAdversary,
        "different": DifferentAdversary,
        "joint": JointAdversary,
        "fixed_x": AdversaryFixedX,
        "fixed_y": AdversaryFixedY,
        "ae": AdversaryAe
    }

    return adversary_functions[adversary_type]


class AdversaryAe(nn.Module):
    pass


class AdversaryCNN(nn.Module):
    def __init__(self, size_x, channel_x, dim_adv, final_activation="none"):
        super(AdversaryCNN, self).__init__()

        self.size_x = size_x
        self.dim_h = 64

        self.cnn = nn.Sequential(
            nn.Conv2d(channel_x, self.dim_h, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2),
            nn.Conv2d(self.dim_h, self.dim_h * 2, 4, 2, 1),
            nn.BatchNorm2d(self.dim_h * 2),
            nn.LeakyReLU(0.2),
        )

        fc_modules = [
            nn.Linear((self.dim_h * 2) * (size_x // 4) * (size_x // 4), 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, dim_adv),
        ]

        if final_activation is None:
            pass
        elif final_activation == "tanh":
            fc_modules.append(nn.Tanh())
        else:
            raise NotImplementedError

        self.fc = nn.Sequential(*fc_modules)

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(-1, (self.dim_h * 2) * (self.size_x // 4) * (self.size_x // 4))
        x = self.fc(x)
        return x


class AdversaryFc(nn.Module):
    def __init__(self, in_dim, out_dim, final_activation="tanh", DIM=8):
        super(AdversaryFc, self).__init__()

        self.fc = nn.Sequential(
            *dense_block(in_dim, DIM, activation="leaky_relu", normalize=True, dropout=False),
            *dense_block(DIM, out_dim, activation=final_activation, normalize=False, dropout=False),
        )

    def forward(self, x):
        return self.fc(x)


class SameAdversary(nn.Module):
    def __init__(self, args, final_activation=None, adversary_type="cnn"):
        super(SameAdversary, self).__init__()

        if adversary_type == "cnn":
            self.adversary = AdversaryCNN(size_x=args.size_x, channel_x=args.channel_x, dim_adv=args.dim_adv, final_activation=final_activation)
        elif adversary_type == "fc":
            self.adversary = AdversaryFc(in_dim=args.dim_x, out_dim=args.dim_adv, final_activation=final_activation)
        else:
            raise NotImplementedError

    def forward(self, x, y):
        return self.adversary(x), self.adversary(y)


class AdversaryFixedX(nn.Module):
    def __init__(self, args, final_activation=None, adversary_type="cnn"):
        super(AdversaryFixedX, self).__init__()

        if adversary_type == "cnn":
            self.adversary = AdversaryCNN(size_x=args.size_y, channel_x=args.channel_y, dim_adv=args.dim_adv, final_activation=final_activation)
        elif adversary_type == "fc":
            self.adversary = AdversaryFc(in_dim=args.dim_y, out_dim=args.dim_adv, final_activation=final_activation)
        else:
            raise NotImplementedError

    def forward(self, x, y):
        return x.view(x.shape[0], -1), self.adversary(y)


class AdversaryFixedY(nn.Module):
    def __init__(self, args, final_activation=None, adversary_type="cnn"):
        super(AdversaryFixedY, self).__init__()

        if adversary_type == "cnn":
            self.adversary = AdversaryCNN(size_x=args.size_x, channel_x=args.channel_x, dim_adv=args.dim_adv, final_activation=final_activation)
        elif adversary_type == "fc":
            self.adversary = AdversaryFc(in_dim=args.dim_x, out_dim=args.dim_adv, final_activation=final_activation)
        else:
            raise NotImplementedError

    def forward(self, x, y):
        return self.adversary(x), y.view(y.shape[0], -1)


class DifferentAdversary(nn.Module):
    def __init__(self, args, final_activation=None, adversary_type="cnn"):
        super(DifferentAdversary, self).__init__()

        if adversary_type == "cnn":
            # TODO: Code this.
            self.adversary_x = AdversaryCNN(size_x=args.size_x, channel_x=args.channel_x, dim_adv=args.dim_adv, final_activation=final_activation)
            self.adversary_y = AdversaryCNN(size_x=args.size_y, channel_x=args.channel_y, dim_adv=args.dim_adv, final_activation=final_activation)
        elif adversary_type == "fc":
            self.adversary_x = AdversaryFc(in_dim=args.dim_x, out_dim=args.dim_adv, final_activation=final_activation)
            self.adversary_y = AdversaryFc(in_dim=args.dim_y, out_dim=args.dim_adv, final_activation=final_activation)
        else:
            raise NotImplementedError

    def forward(self, x, y):
        return self.adversary_x(x), self.adversary_y(y)


class JointAdversary(nn.Module):
    def __init__(self, args, final_activation=None, adversary_type="cnn"):
        super(JointAdversary, self).__init__()

        if adversary_type == "cnn":
            self.adversary_x = AdversaryCNN(size_x=args.size_x, channel_x=args.channel_x, dim_adv=args.dim_adv, final_activation=final_activation)

            if args.size_x == 32 and args.size_y == 64:
                modules = [
                    nn.Conv2d(args.channel_y, args.channel_x, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(args.channel_x),
                    nn.LeakyReLU(0.2)
                ]
            else:
                modules = [
                    nn.Conv2d(args.channel_y, args.channel_x, 3, 1, 1),
                    nn.LeakyReLU(0.2)
                ]

            self.mapping_yx = nn.Sequential(*modules)
        elif adversary_type == "fc":
            self.adversary_x = AdversaryFc(in_dim=args.dim_x, out_dim=args.dim_adv, final_activation=final_activation)
            self.mapping_yx = nn.Sequential(
                *dense_block(in_dim=args.dim_y, out_dim=16, activation="leaky_relu", normalize=False, dropout=False),
                *dense_block(in_dim=16, out_dim=args.dim_x, activation="leaky_relu", normalize=False, dropout=False)
            )
        else:
            raise NotImplementedError

    def forward(self, x, y):
        f_x = self.adversary_x(x)

        y = self.mapping_yx(y)
        f_y = self.adversary_x(y)

        return f_x, f_y

# class Adversary(nn.Module):
#     def __init__(self, size_x, channel_x, final_activation=None):
#         super(Adversary, self).__init__()
#
#         out_cnn_size = size_x // 4
#         self.out_cnn_size = out_cnn_size
#         dim_adv = size_x * size_x * channel_x
#
#         self.cnn = nn.Sequential(
#             *cnn_block(channel_x, 64, resizing="half", activation="leaky_relu", normalize=False, dropout=False),
#             *cnn_block(64, 128, resizing="half", activation="leaky_relu", normalize=True, dropout=False),
#         )
#
#         self.fc = nn.Sequential(
#             *dense_block(128 * out_cnn_size * out_cnn_size, 1024, activation="leaky_relu", normalize=True, dropout=False),
#             *dense_block(1024, dim_adv, activation=final_activation, normalize=False, dropout=False),
#         )
#
#     def forward(self, x):
#         batch_size = x.shape[0]
#
#         x = self.cnn(x)
#         x = x.view(batch_size, -1)
#         x = self.fc(x)
#         return x
#
#
#
#
# class JointAdversary(nn.Module):
#     def __init__(self, args, final_activation=None):
#         super(JointAdversary, self).__init__()
#
#         self.adversary_x = Adversary(size_x=args.size_x, channel_x=args.channel_x, final_activation=final_activation).cuda()
#
#         if args.size_y == args.size_x * 2:
#             resizing = "double"
#         elif args.size_y == args.size_x:
#             resizing = "same"
#         elif args.size_y * 2 == args.size_x:
#             resizing = "half"
#         else:
#             raise NotImplementedError
#
#         self.mapping_yx = nn.Sequential(
#             *cnn_block(args.channel_y, args.channel_x, resizing=resizing, activation="leaky_relu", normalize=False, dropout=False),
#         )
#
#     def forward(self, x, y):
#         f_x = self.adversary_x(x)
#
#         y = self.mapping_yx(y)
#         f_y = self.adversary_x(y)
#
#         return f_x, f_y
