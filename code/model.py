import torch
import torch.nn as nn
import torch.nn.functional as F


class Waldo_UNet(nn.Module):
    def __init__(self, n_channels, bilinear=True):
        super(Waldo_UNet, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, 2)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=(2, 2), stride=(2, 2))
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))

    def forward(self, x):
        return self.conv(x)


from keras.layers import Activation, Dropout, BatchNormalization, Conv2DTranspose, Deconvolution2D, Input, Conv2D, \
    concatenate, add, Reshape
from keras.models import Model
from keras.regularizers import l2

def relu(x): return Activation('relu')(x)


def dropout(x, p): return Dropout(p)(x) if p else x


def bn(x): return BatchNormalization(axis=-1)(x)


def relu_bn(x): return relu(bn(x))


def concat(xs): return concatenate(xs, axis=-1)


def conv(x, nf, sz, wd, p, stride=1):
    x = Conv2D(nf, kernel_size=sz, strides=stride, padding="same",
               kernel_initializer='he_uniform', kernel_regularizer=l2(wd))(x)
    return dropout(x, p)


def conv_relu_bn(x, nf, sz=3, wd=0, p=0, stride=1):
    return conv(relu_bn(x), nf, sz, wd=wd, p=p, stride=stride)


def dense_block(n, x, growth_rate, p, wd):
    added = []
    for i in range(n):
        b = conv_relu_bn(x, growth_rate, p=p, wd=wd)
        x = concat([x, b])
        added.append(b)
    return x, added


def transition_dn(x, p, wd):
    #     x = conv_relu_bn(x, x.get_shape().as_list()[-1], sz=1, p=p, wd=wd)
    #     return MaxPooling2D(strides=(2, 2))(x)
    return conv_relu_bn(x, x.get_shape().as_list()[-1], sz=1, p=p, wd=wd, stride=2)


def down_path(x, nb_layers, growth_rate, p, wd):
    skips = []
    for i, n in enumerate(nb_layers):
        x, added = dense_block(n, x, growth_rate, p, wd)
        skips.append(x)
        x = transition_dn(x, p=p, wd=wd)
    return skips, added


def transition_up(added, wd=0):
    x = concat(added)
    _, r, c, ch = x.get_shape().as_list()
    return Conv2DTranspose(ch, (3, 3), strides=(2, 2), kernel_initializer='he_uniform',
                           padding='same', kernel_regularizer=l2(wd))(x)


def up_path(added, skips, nb_layers, growth_rate, p, wd):
    for i, n in enumerate(nb_layers):
        x = transition_up(added, wd)
        x = concat([x, skips[i]])
        x, added = dense_block(n, x, growth_rate, p, wd)
    return x


def reverse(a): return list(reversed(a))


def create_tiramisu(nb_classes, img_input, nb_dense_block=6,
                    growth_rate=16, nb_filter=48, nb_layers_per_block=5, p=None, wd=0):
    if type(nb_layers_per_block) is list or type(nb_layers_per_block) is tuple:
        nb_layers = list(nb_layers_per_block)
    else:
        nb_layers = [nb_layers_per_block] * nb_dense_block

    x = conv(img_input, nb_filter, 3, wd, 0)
    skips, added = down_path(x, nb_layers, growth_rate, p, wd)
    x = up_path(added, reverse(skips[:-1]), reverse(nb_layers[:-1]), growth_rate, p, wd)

    x = conv(x, nb_classes, 1, wd, 0)
    _, r, c, f = x.get_shape().as_list()
    x = Reshape((-1, nb_classes))(x)
    return Activation('softmax')(x)
