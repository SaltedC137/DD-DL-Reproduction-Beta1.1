import torch
import torchvision.models as models
import torch.nn as nn
from torch.nn import functional as F

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class double_conv2d_bn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, strides=1, padding=1):
        super(double_conv2d_bn, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=kernel_size,
                               stride=strides, padding=padding, bias=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=kernel_size,
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
        self.conv1 = nn.ConvTranspose2d(in_channels, out_channels,
                                        kernel_size=kernel_size,
                                        stride=strides, bias=True)
        self.bn1 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        return out


class Unet(nn.Module):
    def __init__(self, inchanel, outchanel):
        super(Unet, self).__init__()
        self.layer1_conv = double_conv2d_bn(inchanel, 64)
        self.layer2_conv = double_conv2d_bn(64, 128)
        self.layer3_conv = double_conv2d_bn(128, 256)
        self.layer4_conv = double_conv2d_bn(256, 512)
        self.layer5_conv = double_conv2d_bn(512, 1024)
        self.layer6_conv = double_conv2d_bn(1024, 512)
        self.layer7_conv = double_conv2d_bn(512, 256)
        self.layer8_conv = double_conv2d_bn(256, 128)
        self.layer9_conv = double_conv2d_bn(128, 64)
        self.layer10_conv = nn.Conv2d(64, outchanel, kernel_size=3,
                                      stride=1, padding=1, bias=True)

        self.deconv1 = deconv2d_bn(1024, 512)
        self.deconv2 = deconv2d_bn(512, 256)
        self.deconv3 = deconv2d_bn(256, 128)
        self.deconv4 = deconv2d_bn(128, 64)

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
        concat1 = torch.cat([convt1, conv4], dim=1)
        conv6 = self.layer6_conv(concat1)

        convt2 = self.deconv2(conv6)
        concat2 = torch.cat([convt2, conv3], dim=1)
        conv7 = self.layer7_conv(concat2)

        convt3 = self.deconv3(conv7)
        concat3 = torch.cat([convt3, conv2], dim=1)
        conv8 = self.layer8_conv(concat3)

        convt4 = self.deconv4(conv8)
        concat4 = torch.cat([convt4, conv1], dim=1)
        conv9 = self.layer9_conv(concat4)
        outp = self.layer10_conv(conv9)
        return outp

class PyramidPool(nn.Module):
    def __init__(self, in_channels, out_channels, pool_size):
        super(PyramidPool, self).__init__()
        self.features = nn.Sequential(
            nn.AdaptiveAvgPool2d(pool_size),
            nn.Conv2d(in_channels, out_channels, 1, bias=True),
            nn.BatchNorm2d(out_channels, momentum=0.95),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        size = x.shape
        output = F.upsample_bilinear(self.features(x), size[2:])
        return output


def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


class PSPNet(nn.Module):
    def __init__(self, num_classes, pretrained=False):
        super(PSPNet, self).__init__()
        print("initializing model")

        self.resnet = models.resnet50(pretrained=pretrained)
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.layer5a = PyramidPool(2048, 512, 1)
        self.layer5b = PyramidPool(2048, 512, 2)
        self.layer5c = PyramidPool(2048, 512, 3)
        self.layer5d = PyramidPool(2048, 512, 6)

        self.final = nn.Sequential(
            nn.Conv2d(4096, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512, momentum=.95),
            nn.ReLU(inplace=True),
            nn.Dropout(.1),
            nn.Conv2d(512, num_classes, 1),
        )

        initialize_weights(self.layer5a, self.layer5b, self.layer5c, self.layer5d, self.final)

    def forward(self, x):
        size = x.size()
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.final(torch.cat([
            x,
            self.layer5a(x),
            self.layer5b(x),
            self.layer5c(x),
            self.layer5d(x),
        ], 1))
        return F.upsample_bilinear(x, size[2:])


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rates):
        super(ASPP, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, dilation=dilation_rates[0],
                               padding=dilation_rates[0])
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, dilation=dilation_rates[1],
                               padding=dilation_rates[1])
        self.conv4 = nn.Conv2d(in_channels, out_channels, kernel_size=3, dilation=dilation_rates[2],
                               padding=dilation_rates[2])
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv5 = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        x5 = self.pool(x)
        x5 = self.conv5(x5)
        x5 = torch.nn.functional.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=False)
        return torch.cat((x1, x2, x3, x4, x5), dim=1)


class DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes):
        super(DeepLabV3Plus, self).__init__()
        self.resnet = models.resnet50(pretrained=True)

        self.layer0 = nn.Sequential(self.resnet.conv1, self.resnet.bn1, self.resnet.relu)
        self.layer1 = self.resnet.layer1
        self.layer2 = self.resnet.layer2
        self.layer3 = self.resnet.layer3
        self.layer4 = self.resnet.layer4

        self.aspp = ASPP(in_channels=2048, out_channels=256, dilation_rates=[6, 12, 18])

        self.ca1 = ChannelAttention(in_planes=1280)
        self.sa1 = SpatialAttention()


        self.decoder = nn.Sequential(
            nn.Conv2d(256 * 5, 48, kernel_size=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, num_classes, kernel_size=1)
        )

    def forward(self, x):
        x0 = self.layer0(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        x_aspp = self.aspp(x4)
        x_aspp = self.ca1(x_aspp) * x_aspp
        x_aspp = self.sa1(x_aspp) * x_aspp
        x_decoder = self.decoder(x_aspp)

        x_decoder = torch.nn.functional.interpolate(x_decoder, size=x.size()[2:], mode='bilinear', align_corners=False)
        return x_decoder


class ContinusParalleConv(nn.Module):
    # 一个连续的卷积模块，包含BatchNorm 在前 和 在后 两种模式
    def __init__(self, in_channels, out_channels, pre_Batch_Norm=False):
        super(ContinusParalleConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        if pre_Batch_Norm:
            self.Conv_forward = nn.Sequential(
                nn.BatchNorm2d(self.in_channels),
                nn.ReLU(),
                nn.Conv2d(self.in_channels, self.out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1))

        else:
            self.Conv_forward = nn.Sequential(
                nn.Conv2d(self.in_channels, self.out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1),
                nn.BatchNorm2d(self.out_channels),
                nn.ReLU())

    def forward(self, x):
        x = self.Conv_forward(x)
        return x


class UnetPlusPlus(nn.Module):
    def __init__(self, num_classes, deep_supervision=True):
        super(UnetPlusPlus, self).__init__()
        self.num_classes = num_classes
        self.deep_supervision = deep_supervision
        self.filters = [32, 64, 128, 256, 512]

        self.CONV3_1 = ContinusParalleConv(256 * 2, 256, pre_Batch_Norm=True)

        self.CONV2_2 = ContinusParalleConv(128 * 3, 128, pre_Batch_Norm=True)
        self.CONV2_1 = ContinusParalleConv(128 * 2, 128, pre_Batch_Norm=True)

        self.CONV1_1 = ContinusParalleConv(64 * 2, 64, pre_Batch_Norm=True)
        self.CONV1_2 = ContinusParalleConv(64 * 3, 64, pre_Batch_Norm=True)
        self.CONV1_3 = ContinusParalleConv(64 * 4, 64, pre_Batch_Norm=True)

        self.CONV0_1 = ContinusParalleConv(32 * 2, 32, pre_Batch_Norm=True)
        self.CONV0_2 = ContinusParalleConv(32 * 3, 32, pre_Batch_Norm=True)
        self.CONV0_3 = ContinusParalleConv(32 * 4, 32, pre_Batch_Norm=True)
        self.CONV0_4 = ContinusParalleConv(32 * 5, 32, pre_Batch_Norm=True)

        self.stage_0 = ContinusParalleConv(3, 32, pre_Batch_Norm=False)
        self.stage_1 = ContinusParalleConv(32, 64, pre_Batch_Norm=False)
        self.stage_2 = ContinusParalleConv(64, 128, pre_Batch_Norm=False)
        self.stage_3 = ContinusParalleConv(128, 256, pre_Batch_Norm=False)
        self.stage_4 = ContinusParalleConv(256, 512, pre_Batch_Norm=False)

        self.pool = nn.MaxPool2d(2)

        self.upsample_3_1 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1)

        self.upsample_2_1 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.upsample_2_2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1)

        self.upsample_1_1 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.upsample_1_2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.upsample_1_3 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1)

        self.upsample_0_1 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1)
        self.upsample_0_2 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1)
        self.upsample_0_3 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1)
        self.upsample_0_4 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1)

        # 分割头
        self.final_super_0_1 = nn.Sequential(
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, self.num_classes, 3, padding=1),
        )
        self.final_super_0_2 = nn.Sequential(
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, self.num_classes, 3, padding=1),
        )
        self.final_super_0_3 = nn.Sequential(
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, self.num_classes, 3, padding=1),
        )
        self.final_super_0_4 = nn.Sequential(
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, self.num_classes, 3, padding=1),
        )

    def forward(self, x):
        x_0_0 = self.stage_0(x)
        x_1_0 = self.stage_1(self.pool(x_0_0))
        x_2_0 = self.stage_2(self.pool(x_1_0))
        x_3_0 = self.stage_3(self.pool(x_2_0))
        x_4_0 = self.stage_4(self.pool(x_3_0))

        x_0_1 = torch.cat([self.upsample_0_1(x_1_0), x_0_0], 1)
        x_0_1 = self.CONV0_1(x_0_1)

        x_1_1 = torch.cat([self.upsample_1_1(x_2_0), x_1_0], 1)
        x_1_1 = self.CONV1_1(x_1_1)

        x_2_1 = torch.cat([self.upsample_2_1(x_3_0), x_2_0], 1)
        x_2_1 = self.CONV2_1(x_2_1)

        x_3_1 = torch.cat([self.upsample_3_1(x_4_0), x_3_0], 1)
        x_3_1 = self.CONV3_1(x_3_1)

        x_2_2 = torch.cat([self.upsample_2_2(x_3_1), x_2_0, x_2_1], 1)
        x_2_2 = self.CONV2_2(x_2_2)

        x_1_2 = torch.cat([self.upsample_1_2(x_2_1), x_1_0, x_1_1], 1)
        x_1_2 = self.CONV1_2(x_1_2)

        x_1_3 = torch.cat([self.upsample_1_3(x_2_2), x_1_0, x_1_1, x_1_2], 1)
        x_1_3 = self.CONV1_3(x_1_3)

        x_0_2 = torch.cat([self.upsample_0_2(x_1_1), x_0_0, x_0_1], 1)
        x_0_2 = self.CONV0_2(x_0_2)

        x_0_3 = torch.cat([self.upsample_0_3(x_1_2), x_0_0, x_0_1, x_0_2], 1)
        x_0_3 = self.CONV0_3(x_0_3)

        x_0_4 = torch.cat([self.upsample_0_4(x_1_3), x_0_0, x_0_1, x_0_2, x_0_3], 1)
        x_0_4 = self.CONV0_4(x_0_4)

        if self.deep_supervision:
            out_put1 = self.final_super_0_1(x_0_1)
            out_put2 = self.final_super_0_2(x_0_2)
            out_put3 = self.final_super_0_3(x_0_3)
            out_put4 = self.final_super_0_4(x_0_4)
            return [out_put1, out_put2, out_put3, out_put4]
        else:
            return self.final_super_0_4(x_0_4)






class DeepLabV3Plus2(nn.Module):
    def __init__(self, num_classes):
        super(DeepLabV3Plus2, self).__init__()
        # self.resnet = models.resnet50(pretrained=False)
        self.resnet = models.mobilenet_v2(pretrained=True)

        self.layer0 = nn.Sequential(self.resnet.features[0], self.resnet.features[1])
        self.layer1 = nn.Sequential(self.resnet.features[2], self.resnet.features[3], self.resnet.features[4],
                                    self.resnet.features[5], self.resnet.features[6])
        self.layer2 = nn.Sequential(self.resnet.features[7], self.resnet.features[8], self.resnet.features[9],
                                    self.resnet.features[10])
        self.layer3 = nn.Sequential(self.resnet.features[11], self.resnet.features[12], self.resnet.features[13],
                                    self.resnet.features[14], self.resnet.features[15], self.resnet.features[16],
                                    self.resnet.features[17])
        self.layer4 = nn.Sequential(self.resnet.features[18])


        self.aspp = ASPP(in_channels=1280, out_channels=256, dilation_rates=[6, 12, 18])

        self.decoder = nn.Sequential(
            nn.Conv2d(256 * 5, 48, kernel_size=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, num_classes, kernel_size=1)
        )

    def forward(self, x):
        x0 = self.layer0(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        x_aspp = self.aspp(x4)
        x_decoder = self.decoder(x_aspp)

        x_decoder = torch.nn.functional.interpolate(x_decoder, size=x.size()[2:], mode='bilinear', align_corners=False)
        return x_decoder






class DeepLabV3Plus4(nn.Module):
    def __init__(self, num_classes):
        super(DeepLabV3Plus4, self).__init__()
        self.resnet = models.resnet50(pretrained=False) #要注释
        # self.resnet = models.mobilenet_v2(pretrained=True)

        # self.layer0 = nn.Sequential(self.resnet.features[0], self.resnet.features[1])
        # self.layer1 = nn.Sequential(self.resnet.features[2], self.resnet.features[3], self.resnet.features[4],
        #                             self.resnet.features[5], self.resnet.features[6])
        # self.layer2 = nn.Sequential(self.resnet.features[7], self.resnet.features[8], self.resnet.features[9],
        #                             self.resnet.features[10])
        # self.layer3 = nn.Sequential(self.resnet.features[11], self.resnet.features[12], self.resnet.features[13],
        #                             self.resnet.features[14], self.resnet.features[15], self.resnet.features[16],
        #                             self.resnet.features[17])
        # self.layer4 = nn.Sequential(self.resnet.features[18])

        #修复
        self.layer0 = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu,
            self.resnet.maxpool
        )
        self.layer1 = self.resnet.layer1
        self.layer2 = self.resnet.layer2
        self.layer3 = self.resnet.layer3
        self.layer4 = self.resnet.layer4
        
        self.aspp = ASPP(in_channels=2048, out_channels=256, dilation_rates=[6, 12, 18])
        self.ca1 = ChannelAttention(in_planes=1280)

        
        # self.aspp = ASPP(in_channels=1280, out_channels=256, dilation_rates=[6, 12, 18])
        # self.ca1 = ChannelAttention(in_planes=1280)
        self.sa1 = SpatialAttention()
        self.decoder = nn.Sequential(
            nn.Conv2d(256 * 5, 48, kernel_size=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, num_classes, kernel_size=1)
        )

    def forward(self, x):
        x0 = self.layer0(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x_aspp = self.aspp(x4)
        x_aspp = self.ca1(x_aspp) * x_aspp
        x_aspp = self.sa1(x_aspp) * x_aspp
        x_decoder = self.decoder(x_aspp)
        x_decoder = torch.nn.functional.interpolate(x_decoder, size=x.size()[2:], mode='bilinear', align_corners=False)
        return x_decoder




if __name__ == '__main__':
    # model = DeepLabV3Plus(2)
    # input = torch.randn(size=(1,3,256,256))
    # out = model(input)
    # print(out.shape)

    # model = DeepLabV3Plus2(2)
    # input = torch.randn(size=(1, 3, 256, 256))
    # out = model(input)
    # print(out.shape)

    model = DeepLabV3Plus4(2)
    input = torch.randn(size=(1, 3, 256, 256))
    out = model(input)
    print(out.shape)
