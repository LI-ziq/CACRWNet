import torch.utils.model_zoo as model_zoo
import math
from DeepLab import *
import config



model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3mb4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',

}

# 下面是50、101、152层
class Bottleneck(nn.Module):
    """
    注意：原论文中，在虚线残差结构的主分支上，第一个1x1卷积层的步距是2，第二个3x3卷积层步距是1。
    但在pytorch官方实现过程中是第一个1x1卷积层的步距是1，第二个3x3卷积层步距是2，
    这么做的好处是能够在top1上提升大概0.5%的准确率。
    可参考Resnet v1.5 https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch
    """
    expansion = 4
    """残差结构中最后一层卷积核的个数是与之前的4倍"""

    def __init__(self, in_channel, out_channel, stride=1, downsample=None,
                 groups=1, width_per_group=64):
        super(Bottleneck, self).__init__()

        width = int(out_channel * (width_per_group / 64.)) * groups

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width,
                               kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.bn1 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channel*self.expansion,
                               kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        # self.sge = SpatialGroupEnhance(64)

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out


############################################################
class ResNet_New(nn.Module):
    # 增加全局特征提取器之后的整体网络框架

    def __init__(self,
                 block,  # 选择两者中的一个
                 blocks_num,  # 使用残差结构的数目，是一个列表参数
                 num_classes=1000,  # 训练集的分类数
                 include_top=True,  # 方便在resNet的基础上搭建更复杂的网络
                 groups=1,
                 width_per_group=64,
                 look=True):
        super(ResNet_New, self).__init__()
        self.include_top = include_top # 将include.top传入类变量
        self.in_channel = 64 #定义输入特征矩阵的深度，Maxpool之后的深度
        self.look = look

        self.groups = groups
        self.width_per_group = width_per_group

        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        # 定义ResNet+GEF中的第一个卷积层（输入图像的深度，卷积核个数，卷积核大小，步长，补零）输入图像是n×n×3，输出是112×112×64
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 定义第二层最大池化层（池化核大小，步长，补零）输入为112×112×64，输出为56×56×64
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        # 模块1
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        # self.layer33 = self._make_layer123(Bottleneck, 256, 2, stride=2)


        #  #########################降维###############
        self.R5 = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)
        self.R4 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.R3 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        # self.R2 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        # self.R1 = nn.Conv2d(64, 256, kernel_size=1, stride=1, padding=0)


        #  #########################金字塔 ###############
        ###########################################################
        self.SEFPN1 = SE_FPN(256, 256)
        self.SEFPN2 = SE_FPN(256, 256)
        self.SEFPN3 = SE_FPN(256, 256)
        # Smooth layers
        self.smooth0 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        #  ######################### V型网络 ###############
        # ###########################################################

        self.RSU1 = RSU4F2(256, 256, 256)

        self.SE = SE_Block(256)
        # self.red = reduction(256, 256)

        # #  ####################
        # ###################################
        # ###########################################################
        #
        self.max1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # # self.SGE = Reduction_SGE(256, 256)
        # # 将输出的图像长和宽转化为一样的大小
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.down7 = nn.AdaptiveAvgPool2d((7, 7))
        # self.down14 = nn.AdaptiveAvgPool2d((14, 14))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        # 对卷积层进行初始化操作

    def _make_layer(self, block, channel, block_num, stride=1):
        #    定义makelayer函数（basicblock或者bottleneck，残差结构中第一层的卷积核个数，该层包含的残差结构数目，步长）
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            # 18层和34层跳过此段
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))
            # 对于模块1的第一层，只需要改变大小，不需要改变深度；而模块2、3、4的第一层既需要改变大小，又需要改变深度

        layers = []
        layers.append(block(self.in_channel,
                            channel,
                            downsample=downsample,
                            stride=stride,))
                            # groups=self.groups,
                            # width_per_group=self.width_per_group))
        self.in_channel = channel * block.expansion
        # 这是残差块中的第一个结构，是需要虚线的
        for _ in range(1, block_num):
            layers.append(block(self.in_channel,
                                channel,))
                                # groups=self.groups,
                                # width_per_group=self.width_per_group))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
   # 56×56×64
        x = self.layer1(x)
        # x = self.SE1(x)
        # x = self.sge(x)
        c2 = x       # 56×56×256
        x = self.layer2(x)
        # x = self.sge(x)
        # x = self.SE2(x)
        c3 = x       # 28×28×512
        x = self.layer3(x)
        # x = self.sge(x)
        # x = self.SE3(x)
        c4 = x       # 14×14×1024
        # print(c4.size())
        x = self.layer4(x)
        # x = self.SE(x)
        # x = self.sge(x)
        c5 = x       # 7×7×2048

# ####################### 降维模块 #######################
        d2 = c2           # 56×56×256
        d3 = self.R3(c3)  # 28×28×256
        d4 = self.R4(c4)  # 14×14×256
        d5 = self.R5(c5)  # 7×7×256

        ret_reduction = d2

# ####################### 金字塔模型 #######################
#
        U1 = d5                   # 7×7×256
        U2 = self.SEFPN1(U1, d4)  # 14×14×256
        U3 = self.SEFPN2(U2, d3)  # 28×28×256
        U4 = self.SEFPN3(U3, d2)  # 56×56×256

        ret_FPN = U4

        U1 = self.smooth0(U1)    # 7×7×256
        U2 = self.smooth1(U2)    # 14×14×256
        U3 = self.smooth2(U3)    # 28×28×256
        U4 = self.smooth3(U4)   # 56×56×256

#
# # ####################### V型网络 #######################

        d2 = U4
        d3 = U3
        d4 = U2
        d5 = U1

        ret_before = d2

        d2 = self.RSU1(d2)  # 56×56×256
        ret_after = d2
        if self.look:
            return ret_before, ret_after,ret_reduction, ret_FPN

        d3 = self.RSU1(d3)  # 28×28×256


        d4 = self.RSU1(d4)  # 14×1R×256

        d5 = self.RSU1(d5)  # 7×7×256

# # #  #################### 注意力机制 #######################

        d3 = d3 + self.max1(d2)
        d3 = self.smooth0(d3)

        d4 = d4 + self.max1(d3)
        d4 = self.smooth1(d4)


        d5 = d5 + self.max1(d4)
        d5 = self.smooth2(d5)

        d2 = self.SE(d2)
        d3 = self.SE(d3)
        d4 = self.SE(d4)
        d5 = self.SE(d5)

        # #
        # d3 = self.red(d3)
        # d4 = self.red(d4)
        # d5 = self.red(d5)



# ####################### 转换大小 #######################

        out22 = self.down7(d2)
        out33 = self.down7(d3)
        out44 = self.down7(d4)
        # out55 = self.down7(d5)
        out55 = d5
# ####################### 全连接层 #######################
        out = torch.cat((out22, out33, out44, out55), 1)
        out = self.avg_pool(out)
        out = torch.flatten(out, 1)
        # print(out.shape)
        return out


def resnext50_Deeplab(pretrained=False, num_classes=1000, include_top=True, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    # groups = 32
    # width_per_group = 4
    # model = ResNet_New(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top,
    #                    groups=groups, width_per_group=width_per_group)
    model = ResNet_New(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)
    if pretrained:
        pretrained_model = model_zoo.load_url(model_urls['resnet50'])
        state = model.state_dict()
        for key in state.keys():
            if key in pretrained_model.keys():
                if "fc" not in key and "features.13" not in key:
                    state[key] = pretrained_model[key]
        model.load_state_dict(state)
    return model


class resnet50_DV(nn.Module):
    # 重新以resnet50为主干构建自己的全连接层
    def __init__(self):
        super(resnet50_DV, self).__init__()
        self.backbone = self._get_backbone()
        self.fc = nn.Linear(1024, config.NUM_CLASSES)

    def _get_backbone(self):
        backbone = resnext50_Deeplab(pretrained=True, num_classes=config.NUM_CLASSES)
        return backbone

    def forward(self, x):
        ret_before, ret_after, ret_reduction, ret_FPN = self.backbone(x)
        # out = out.view(-1, 1024)    #   展平为一维
        # out = self.fc(out)
        return ret_before, ret_after, ret_reduction, ret_FPN


