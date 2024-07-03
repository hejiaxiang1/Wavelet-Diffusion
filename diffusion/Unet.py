import torch
import torch.nn as nn
import torch.nn.functional as F

class double_conv2d_bn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, strides=1, padding=1):
        super(double_conv2d_bn, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                               stride=strides, padding=padding, bias=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size,
                               stride=strides, padding=padding, bias=True)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        return out

class deconv2d_bn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, strides=2):
        super(deconv2d_bn, self).__init__()
        self.conv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size,
                                        stride=strides, bias=True)
        self.bn1 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        return out

class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()
        self.layer1_conv = double_conv2d_bn(3, 8)
        self.layer2_conv = double_conv2d_bn(8, 16)
        self.layer3_conv = double_conv2d_bn(16, 32)
        self.layer4_conv = double_conv2d_bn(32, 64)
        self.layer5_conv = double_conv2d_bn(64, 128)
        self.layer6_conv = double_conv2d_bn(128, 64)
        self.layer7_conv = double_conv2d_bn(64, 32)
        self.layer8_conv = double_conv2d_bn(32, 16)
        self.layer9_conv = double_conv2d_bn(16, 8)
        self.layer10_conv = nn.Conv2d(8, 3, kernel_size=3, stride=1, padding=1, bias=True)

        self.deconv1 = deconv2d_bn(128, 64)
        self.deconv2 = deconv2d_bn(64, 32)
        self.deconv3 = deconv2d_bn(32, 16)
        self.deconv4 = deconv2d_bn(16, 8)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        conv1 = self.layer1_conv(x)
        pool1 = F.max_pool2d(conv1, 2)

        conv2 = self.layer2_conv(pool1)
        pool2 = F.max_pool2d(conv2, 2)

        conv3 = self.layer3_conv(pool2)
        pool3 = F.max_pool2d(conv3, 2)

        conv4 = self.layer4_conv(pool3)
        pool4 = F.max_pool2d(conv4, 2)

        conv5 = self.layer5_conv(pool4)

        convt1 = self.deconv1(conv5)
        convt1 = self._crop_and_concat(convt1, conv4)

        conv6 = self.layer6_conv(convt1)

        convt2 = self.deconv2(conv6)
        convt2 = self._crop_and_concat(convt2, conv3)

        conv7 = self.layer7_conv(convt2)

        convt3 = self.deconv3(conv7)
        convt3 = self._crop_and_concat(convt3, conv2)

        conv8 = self.layer8_conv(convt3)

        convt4 = self.deconv4(conv8)
        convt4 = self._crop_and_concat(convt4, conv1)

        conv9 = self.layer9_conv(convt4)

        outp = self.layer10_conv(conv9)
        outp = self.sigmoid(outp)

        return outp

    def _crop_and_concat(self, upsampled, bypass):
        _, _, H, W = upsampled.size()
        _, _, H_bypass, W_bypass = bypass.size()
        crop_h = (H_bypass - H) // 2
        crop_w = (W_bypass - W) // 2
        bypass = bypass[:, :, crop_h : crop_h + H, crop_w : crop_w + W]
        return torch.cat((upsampled, bypass), 1)