from typing import Dict, List
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.nn.parameter import Parameter

from torch.nn import Module, Sequential, Conv2d, ReLU, AdaptiveMaxPool2d, AdaptiveAvgPool2d, \
    NLLLoss, BCELoss, CrossEntropyLoss, AvgPool2d, MaxPool2d, Parameter, Linear, Sigmoid, Softmax, Dropout, Embedding

#################金字塔网络##########################

def upsample(in_channels, out_channels, mode='transpose'):
        if mode == 'transpose':
            return nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=2,
                stride=2
            )
        elif mode == 'bilinear':
            return nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                # conv1x1(in_channels, out_channels)
            )
        else:
            return nn.Sequential(
                nn.Upsample(mode='nearest', scale_factor=2),
                # conv1x1(in_channels, out_channels)
            )


def downsample(in_channels, out_channels):
        return nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=2, padding=1)


class SE_FPN(nn.Module):
    def __init__(self, in_channels, out_channels, mode='transpose'):
        super(SE_FPN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mode = mode
        self.upsample = upsample(self.in_channels, self.out_channels, self.mode)

    def forward(self, high, low):
        h = self.upsample(high)
        l = low
        out = h + l
        return out

class FAN_FPN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FAN_FPN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.downsample =downsample(self.in_channels, self.out_channels)

    def forward(self, high, low):
        l = self.downsample(low)
        h = high
        out = h + l
        return out

# #############################注意力网络##########################


# model = reduction(3, 3)
# inputs = torch.rand(1, 3, 28, 28)
# outputs = model(inputs)


class SpatialGroupEnhance(nn.Module):
    def __init__(self, groups = 64):
        super(SpatialGroupEnhance, self).__init__()
        self.groups   = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.weight   = Parameter(torch.zeros(1, groups, 1, 1))
        self.bias     = Parameter(torch.ones(1, groups, 1, 1))
        self.sig      = nn.Sigmoid()

    def forward(self, x): # (b, c, h, w)
        b, c, h, w = x.size()
        x = x.view(b * self.groups, -1, h, w)
        xn = x * self.avg_pool(x)
        xn = xn.sum(dim=1, keepdim=True)
        t = xn.view(b * self.groups, -1)
        t = t - t.mean(dim=1, keepdim=True)
        std = t.std(dim=1, keepdim=True) + 1e-5
        t = t / std
        t = t.view(b, self.groups, h, w)
        t = t * self.weight + self.bias
        t = t.view(b * self.groups, 1, h, w)
        x = x * self.sig(t)
        x = x.view(b, c, h, w)
        return x



# #############################V型网络##########################
class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv, self).__init__(
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


class Down(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__(
            nn.MaxPool2d(2, stride=2),
            DoubleConv(in_channels, out_channels)
        )


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # [N, C, H, W]
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        # padding_left, padding_right, padding_top, padding_bottom
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 bilinear: bool = True,
                ):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.bilinear = bilinear
        self.down1 = Down(in_channels, out_channels)
        self.down2 = Down(in_channels, out_channels)
        self.up1 = Up(in_channels * 2, out_channels, bilinear)
        self.up2 = Up(in_channels * 2, out_channels, bilinear)
        # self.avg = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x1 = x
        x2 = self.down1(x1)

        x3 = self.down2(x2)

        # x3 =  x3 * self.avg(x3)

        x = self.up1(x3, x2)

        # x = x * self.avg(x)

        x = self.up2(x, x1)

        # x = x * self.avg(x)

        return x


class UNet2(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 bilinear: bool = True,
                ):
        super(UNet2, self).__init__()
        self.in_channels = in_channels
        self.bilinear = bilinear
        self.down1 = Down(in_channels, out_channels)
        self.up1 = Up(in_channels * 2, out_channels, bilinear)
        # self.avg = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x1 = x
        x2 = self.down1(x1)
        x = self.up1(x2, x1)

        return x



# #############################膨胀卷积的V型网络##########################


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()

        padding = kernel_size // 2 if dilation == 1 else dilation
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation, bias=False)

        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        # x = x * self.avg(x)
        x = self.bn(x)
        x = self.relu(x)

        return x

#
# class Avg(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.avg = nn.AdaptiveAvgPool2d(1)
#
#     def forward(self, x):
#         x = x * self.avg(x)
#         return x


class RSU4F(nn.Module):
    def __init__(self, in_ch: int, mid_ch: int, out_ch: int):
        super().__init__()
        self.conv_in = ConvBNReLU(in_ch, out_ch)
        self.encode_modules = nn.ModuleList([ConvBNReLU(out_ch, mid_ch),
                                             ConvBNReLU(mid_ch, mid_ch, dilation=2),
                                             ConvBNReLU(mid_ch, mid_ch, dilation=4),

                                             ])

        self.decode_modules = nn.ModuleList([
                                             ConvBNReLU(mid_ch * 2, mid_ch, dilation=2),
                                             ConvBNReLU(mid_ch * 2, out_ch),
                                             ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = self.conv_in(x)

        x = x_in
        encode_outputs = []
        for m in self.encode_modules:
            x = m(x)
            encode_outputs.append(x)

        x = encode_outputs.pop()
        for m in self.decode_modules:
            x2 = encode_outputs.pop()
            x = m(torch.cat([x, x2], dim=1))

        return x + x_in



class RSU4F2(nn.Module):
    def __init__(self, in_ch: int, mid_ch: int, out_ch: int):
        super().__init__()
        self.encode_modules = nn.ModuleList([ConvBNReLU(in_ch, mid_ch),
                                             ConvBNReLU(mid_ch, mid_ch, dilation=2),
                                             ])

        self.decode_modules = nn.ModuleList([
                                             ConvBNReLU(mid_ch * 2, out_ch),
                                             ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = x
        encode_outputs = []
        for m in self.encode_modules:
            x = m(x)
            encode_outputs.append(x)

        x = encode_outputs.pop()
        for m in self.decode_modules:
            x2 = encode_outputs.pop()
            x = m(torch.cat([x, x2], dim=1))

        return x + x_in
# model = RSU4F(256, 256, 256)
#
# inputs = torch.rand(1, 256, 28, 28)
# outputs = model(inputs)

class AtrousBasicBlock(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(AtrousBasicBlock, self).__init__()
        self.inplanes = inplanes
        self.conv1 = nn.Conv2d(inplanes, 64, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=3, dilation=3)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, outplanes, kernel_size=3, stride=1, padding=5, dilation=5)
        self.bn3 = nn.BatchNorm2d(outplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0)
        self.bn4 = nn.BatchNorm2d(outplanes)

    def forward(self, x):
        identify = x
        identify = self.conv4(identify)
        identify = self.bn4(identify)
        identify = self.relu(identify)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identify
        out = self.relu(out)
        return out

class BasicBlock(nn.Module):
    def __init__(self,inplanes,outplanes):
        super(BasicBlock,self).__init__()
        self.inplanes = inplanes
        self.conv1 = nn.Conv2d(inplanes,64,kernel_size=1,stride=1,padding=0)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64,outplanes,kernel_size=1,stride=1,padding=0)
        self.bn3 = nn.BatchNorm2d(outplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(inplanes,outplanes,kernel_size=1,stride=1,padding=0)
        self.bn4 = nn.BatchNorm2d(outplanes)

    def forward(self,x):
        identify = x
        identify = self.conv4(identify)
        identify = self.bn4(identify)
        identify = self.relu(identify)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identify
        out = self.relu(out)
        return out

#  ###############################注意力#############################

class PAM_Module(Module):
    """ Position attention module"""

    # Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        # 先经过3个卷积层生成3个新特征图B C D （尺寸不变）
        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))  # α尺度系数初始化为0，并逐渐地学习分配到更大的权重

        self.softmax = Softmax(dim=-1)  # 对每一行进行softmax

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B × C × H × W)
            returns :
                out : attention value + input feature
                attention: B × (H×W) × (H×W)
        """
        m_batchsize, C, height, width = x.size()
        # B -> (N,C,HW) -> (N,HW,C)
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        # C -> (N,C,HW)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        # BC，空间注意图 -> (N,HW,HW)
        energy = torch.bmm(proj_query, proj_key)
        # S = softmax(BC) -> (N,HW,HW)
        attention = self.softmax(energy)
        # D -> (N,C,HW)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)
        # DS -> (N,C,HW)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # torch.bmm表示批次矩阵乘法
        # output -> (N,C,H,W)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out


class CAM_Module(Module):
    """ Channel attention module"""

    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.gamma = Parameter(torch.zeros(1))  # β尺度系数初始化为0，并逐渐地学习分配到更大的权重
        self.softmax = Softmax(dim=-1)  # 对每一行进行softmax

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B × C × H × W)
            returns :
                out : attention value + input feature
                attention: B × C × C
        """
        m_batchsize, C, height, width = x.size()
        # A -> (N,C,HW)
        proj_query = x.view(m_batchsize, C, -1)
        # A -> (N,HW,C)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        # 矩阵乘积，通道注意图：X -> (N,C,C)
        energy = torch.bmm(proj_query, proj_key)
        # 这里实现了softmax用最后一维的最大值减去了原始数据，获得了一个不是太大的值
        # 沿着最后一维的C选择最大值，keepdim保证输出和输入形状一致，除了指定的dim维度大小为1
        # expand_as表示以复制的形式扩展到energy的尺寸
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy

        attention = self.softmax(energy_new)
        # A -> (N,C,HW)
        proj_value = x.view(m_batchsize, C, -1)
        # XA -> （N,C,HW）
        out = torch.bmm(attention, proj_value)
        # output -> (N,C,H,W)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out


class DANetHead(nn.Module):
    def __init__(self, in_channels,  norm_layer):
        super(DANetHead, self).__init__()
        inter_channels = in_channels  # in_channels=2018，通道数缩减为512

        # self.conv5a = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
        #                             norm_layer(inter_channels), nn.ReLU())
        # self.conv5c = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
        #                             norm_layer(inter_channels), nn.ReLU())

        self.sa = PAM_Module(inter_channels)  # 空间注意力模块
        self.sc = CAM_Module(inter_channels)  # 通道注意力模块

        self.conv51 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels), nn.ReLU())
        self.conv52 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels), nn.ReLU())

        # nn.Dropout2d(p,inplace)：p表示将元素置0的概率；inplace若设置为True，会在原地执行操作。
        # self.conv6 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(256, out_channels, 1))  # 输出通道数为类别的数目
        # self.conv7 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(256, out_channels, 1))
        # self.conv8 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(256, out_channels, 1))

    def forward(self, x):
        # 经过一个1×1卷积降维后，再送入空间注意力模块
        # feat1 = self.conv5a(x)
        sa_feat = self.sa(x)
        # 先经过一个卷积后，再使用有dropout的1×1卷积输出指定的通道数
        sa_conv = self.conv51(sa_feat)
        # sa_output = self.conv6(sa_conv)

        # 经过一个1×1卷积降维后，再送入通道注意力模块
        # feat2 = self.conv5c(x)
        sc_feat = self.sc(x)
        # 先经过一个卷积后，再使用有dropout的1×1卷积输出指定的通道数
        sc_conv = self.conv52(sc_feat)
        # sc_output = self.conv7(sc_conv)

        feat_sum = sa_conv + sc_conv  # 两个注意力模块结果相加
        # sasc_output = self.conv8(feat_sum)  # 最后再送入1个有dropout的1×1卷积中

        # output = [sasc_output]
        # output.append(sa_output)
        # output.append(sc_output)
        return feat_sum  # 输出模块融合后的结果，以及两个模块各自的结果




class SE_Block(nn.Module):
    def __init__(self, ch_in):
        super(SE_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局自适应池化
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in, bias=False),
            nn.ReLU(inplace=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)  # squeeze操作
        y = self.fc(y).view(b, c, 1, 1)  # FC获取通道注意力权重，是具有全局信息的
        return x * y.expand_as(x)  # 注意力作用每一个通道上

# model = DANetHead(512, 512, nn.BatchNorm2d)
#
# inputs = torch.rand(1, 512, 28, 28)
# outputs = model(inputs)

class ChannelAttention(nn.Module):
    def __init__(self, channel):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # 共享权重的MLP
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // 16, False),
            nn.ReLU(),
            nn.Linear(channel // 16, channel, False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        avg_out = self.avg_pool(x).view([b, c])
        max_out = self.max_pool(x).view(b, c)
        avgfc = self.fc(avg_out)
        maxfc = self.fc(max_out)
        out = avgfc + maxfc
        out = self.sigmoid(out).view(b,c,1,1)
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = 7 // 2
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(out)
        return self.sigmoid(out)


class Cbam(nn.Module):
    def __init__(self, channel, kernel_size=7):
        super(Cbam, self).__init__()
        self.channel = ChannelAttention(channel)
        self.spatial = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.channel(x)
        result = out * self.spatial(out)
        return result



class reduction(nn.Module):
    """
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self,inchannel,outchannel):
        super(reduction, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()
        self.bn = nn.BatchNorm2d(outchannel)
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()
        self.conv3_1 = nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Sequential(
            nn.Linear(inchannel, outchannel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()
        p = self.avg_pool(x).view(b, c)
        p = self.fc(p).view(b, 256, 1, 1)
        q = self.conv1(x)
        q = self.bn(q)
        q = self.relu(q)
        q = q * p.expand_as(q)
        k = self.conv3_1(x)
        k = self.relu(k)
        y = q+k
        y = self.conv3_2(y)
        y = self.bn(y)
        y = self.relu(y)
        return y


# model = Cbam(256)
# print(model)
# inputs = torch.rand(1, 256, 28, 28)
# outputs = model(inputs)

class ContextBlock(nn.Module):
    def __init__(self, inplanes, ratio, pooling_type='att',
                 fusion_types=('channel_add',)):
        super(ContextBlock, self).__init__()
        valid_fusion_types = ['channel_add', 'channel_mul']

        assert pooling_type in ['avg', 'att']
        assert isinstance(fusion_types, (list, tuple))
        assert all([f in valid_fusion_types for f in fusion_types])
        assert len(fusion_types) > 0, 'at least one fusion should be used'

        self.inplanes = inplanes
        self.ratio = ratio
        self.planes = int(inplanes * ratio)
        self.pooling_type = pooling_type
        self.fusion_types = fusion_types

        if pooling_type == 'att':
            self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if 'channel_add' in fusion_types:
            self.channel_add_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),  # yapf: disable
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_add_conv = None
        if 'channel_mul' in fusion_types:
            self.channel_mul_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),  # yapf: disable
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_mul_conv = None

    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        if self.pooling_type == 'att':
            input_x = x
            # [N, C, H * W]
            input_x = input_x.view(batch, channel, height * width)
            # [N, 1, C, H * W]
            input_x = input_x.unsqueeze(1)
            # [N, 1, H, W]
            context_mask = self.conv_mask(x)
            # [N, 1, H * W]
            context_mask = context_mask.view(batch, 1, height * width)
            # [N, 1, H * W]
            context_mask = self.softmax(context_mask)
            # [N, 1, H * W, 1]
            context_mask = context_mask.unsqueeze(-1)
            # [N, 1, C, 1]
            context = torch.matmul(input_x, context_mask)
            # [N, C, 1, 1]
            context = context.view(batch, channel, 1, 1)
        else:
            # [N, C, 1, 1]
            context = self.avg_pool(x)
        return context

    def forward(self, x):
        # [N, C, 1, 1]
        context = self.spatial_pool(x)
        out = x
        if self.channel_mul_conv is not None:
            # [N, C, 1, 1]
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
            out = out * channel_mul_term
        if self.channel_add_conv is not None:
            # [N, C, 1, 1]
            channel_add_term = self.channel_add_conv(context)
            out = out + channel_add_term

        return out