import torch
import torch.nn as nn
from torch.optim import lr_scheduler

from utils import spectral_norm as SpectralNorm

# 以图像的对抗生成样本为例,目标类别的 跨模态样本和标签 (实际上就是文本模态的信息)
# LabelNet
class LabelNet(nn.Module):
    def __init__(self, bit, num_classes):
        super(LabelNet, self).__init__()
        self.curr_dim = 16
        self.size = 32
        self.feature = nn.Sequential(nn.Linear(num_classes, 4096), nn.ReLU(True), nn.Linear(4096, self.curr_dim * self.size * self.size))
        conv2d = [
            nn.Conv2d(16, 32, 4, 2, 1),
            # 按照实例归一化
            nn.InstanceNorm2d(32),#和batchnorm不同
            nn.Tanh(),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.InstanceNorm2d(64),
            nn.Tanh()
        ]
        self.conv2d = nn.Sequential(*conv2d)
    def forward(self, label_feature):
        # 标签 特征处理，经过线性，relu，线性
        label_feature = self.feature(label_feature)
        # (手动调整size) view()相当于reshape、resize,重新调整Tensor的形状
        label_feature = label_feature.view(label_feature.size(0), self.curr_dim, self.size, self.size)
        # 二维卷积
        label_feature = self.conv2d(label_feature)
        return label_feature


# TextNet
class TextNet(nn.Module):
    def __init__(self, dim_text, bit, num_classes):
        super(TextNet, self).__init__()
        self.curr_dim = 16
        self.size = 32
        self.feature = nn.Sequential(nn.Linear(dim_text, 4096), nn.ReLU(True), nn.Linear(4096, self.curr_dim * self.size * self.size))
        conv2d = [
            # (输入通道数,输出通道数,卷积核数 kernel_size=1, 步长stride=1, padding=0)
            nn.Conv2d(16, 32, 4, 2, 1),
            nn.InstanceNorm2d(32),
            nn.Tanh(),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.InstanceNorm2d(64),
            nn.Tanh()
        ]
        self.conv2d = nn.Sequential(*conv2d)
    #     文本和标签类似也是相似的处理方法
    def forward(self, text_feature):
        text_feature = self.feature(text_feature)
        # 重新调整张量的形状
        text_feature = text_feature.view(text_feature.size(0), self.curr_dim, self.size, self.size)
        text_feature = self.conv2d(text_feature)
        return text_feature


# Progressive Fusion Module
# 进步的融合模块
# 通过progressive注意力机制,利用多样的目标语义生成细粒度目标语义.
# 原型网络
class PrototypeNet(nn.Module):
    def __init__(self, dim_text, bit, num_classes, channels=64, r=4):
        super(PrototypeNet, self).__init__()

        self.labelnet = LabelNet(bit, num_classes)
        self.textnet = TextNet(dim_text, bit, num_classes)
        # 中间通道数
        inter_channels = int(channels // r)

        # 局部和全局注意力模块
        self.local_att = nn.Sequential(
            # 卷积，BN归一化 ReLU 卷积，BN归一化
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            # normalization方法简化计算过程；经过规范化处理后让数据尽可能保留原始的表达能力
            # 使得网络中的每层输入分布相对稳定，加速模型学习速度，使得参数不那么敏感，允许使用饱和性激活函数sigmoid或者tanh
            # 可以在mini-batch上计算 每个维度的均值和标准差。
            nn.BatchNorm2d(inter_channels),#参数inter_channels为输入数据的shape
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        self.global_att = nn.Sequential(
            # 全局注意力
            # 自适应平均池化 卷积 BN ReLU 卷积 BN
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        self.local_att2 = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        self.global_att2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        # 激活函数
        self.sigmoid = nn.Sigmoid()
        # 卷积
        self.conv2d = nn.Sequential(
            # (输入通道数,输出通道数,卷积核数 kernel_size=1, 步长stride=1, padding=0)
            nn.Conv2d(64, 128, 4, 2, 1),
            # IN适用于小批量并且每个像素信息都必须单独考虑的场景
            nn.InstanceNorm2d(128),
            nn.Tanh(),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.InstanceNorm2d(256),
            nn.Tanh()
        )

        self.hashing = nn.Sequential(nn.Linear(4096, bit), nn.Tanh())
        self.classifier = nn.Sequential(nn.Linear(4096, num_classes), nn.Sigmoid())
    def forward(self, label_feature, text_feature):
        # 原型学习
        # progressive融合模块
        # 获取标签和文本特征
        label_feature = self.labelnet(label_feature)
        text_feature = self.textnet(text_feature)
        # 标签特征和文本特征相加
        xa = label_feature + text_feature
        # 局部和全局注意力
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        # 按照元素对局部和全局注意力相加
        xlg = xl + xg
        # sigmoid激活
        wei = self.sigmoid(xlg)
        # 与初试标签特征和文本特征做 乘法 再求和
        xi = label_feature * wei + text_feature * (1 - wei)

        #类似的步骤再进行一次注意力计算
        xl2 = self.local_att2(xi)
        xg2 = self.global_att2(xi)
        xlg2 = xl2 + xg2
        wei2 = self.sigmoid(xlg2)
        mixed_feature = label_feature * wei2 + text_feature * (1 - wei2)
        # 卷积
        mixed_tensor = self.conv2d(mixed_feature)
        # 重新定义张量的形状 Flatten
        mixed_tensor = mixed_tensor.view(mixed_tensor.size(0), -1)
        # ？？？重构标签和原型码嘛？
        # 求哈希码
        mixed_hashcode = self.hashing(mixed_tensor)
        # 分类器
        mixed_label = self.classifier(mixed_tensor)

        return mixed_feature, mixed_hashcode, mixed_label
# 语义自适应网络体现在？？？


# Semantic Translator
# 语义翻译器
# 在对抗性生成过程中，语义翻译器对 原型融合模块生成的 目标语义 进行翻译（逆卷积），
# 以生成与 受攻击模态同构 的嵌入语义。
class SemanticTranslator(nn.Module):
    def __init__(self):
        super(SemanticTranslator, self).__init__()

        transform = [
            # 使用三次逆卷积完成翻译，也可以使用上采样(保证k=stride,stride即上采样倍数)
            # 输入信号的通道数64，卷积产生通道数48，卷积核大小3，卷积步长1，边补充2，是否添加偏置
            # 还原输入信号，但是只能还原原来输入的shape 其value值是不一样的。
            nn.ConvTranspose2d(64, 48, kernel_size=3, stride=1, padding=2, bias=False),

            nn.InstanceNorm2d(48, affine=False),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(48, 12, kernel_size=6, stride=4, padding=1, bias=False),

            nn.InstanceNorm2d(12, affine=False),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(12, 1, kernel_size=6, stride=4, padding=1, bias=False),
            nn.InstanceNorm2d(1, affine=False),
            nn.ReLU(inplace=True)
            ]
        self.transform = nn.Sequential(*transform)
    def forward(self, label_feature):
        label_feature = self.transform(label_feature)
        return label_feature


# Generator
# 生成器
# 对抗性示例生成器以 良性示例（没有受到攻击） 和 语义翻译网络生成的嵌入语义 作为输入来生成对抗性示例。
# 使用一个Unet网络实现，先编码，然后解码
#
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # 语义翻译网络将通过注意力机制的目标标签和目标示例得到的目标语义 翻译成 嵌入语义
        self.translator = SemanticTranslator()
        # Image Encoder
        curr_dim = 64
        # 准备工作
        self.preprocess = nn.Sequential(
            # 降维，7×7矩阵
            nn.Conv2d(6, curr_dim, kernel_size=7, stride=1, padding=3, bias=True),
            nn.InstanceNorm2d(curr_dim),
            nn.ReLU(inplace=True))
        # 第一层卷积
        self.firstconv = nn.Sequential(
            nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(curr_dim * 2),
            nn.ReLU(inplace=True))
        # 第二卷积
        self.secondconv = nn.Sequential(
            nn.Conv2d(curr_dim * 2, curr_dim * 4, kernel_size=4, stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(curr_dim * 4),
            nn.ReLU(inplace=True))
        # Residual Block
        self.residualblock = nn.Sequential(
            # (self, dim_in, dim_out, net_mode=None):
            ResidualBlock(dim_in=curr_dim * 4, dim_out=curr_dim * 4, net_mode='t'),
            ResidualBlock(dim_in=curr_dim * 4, dim_out=curr_dim * 4, net_mode='t'),
            ResidualBlock(dim_in=curr_dim * 4, dim_out=curr_dim * 4, net_mode='t'),
            ResidualBlock(dim_in=curr_dim * 4, dim_out=curr_dim * 4, net_mode='t'),
            ResidualBlock(dim_in=curr_dim * 4, dim_out=curr_dim * 4, net_mode='t'),
            ResidualBlock(dim_in=curr_dim * 4, dim_out=curr_dim * 4, net_mode='t'))
        # Image Decoder
        # 第一逆卷积
        self.firstconvtrans = nn.Sequential(
            nn.ConvTranspose2d(curr_dim * 4, curr_dim * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(curr_dim * 2),
            nn.ReLU(inplace=True))
        # 第二逆卷积
        self.secondconvtrans = nn.Sequential(
            nn.Conv2d(curr_dim * 4, curr_dim * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(curr_dim * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(curr_dim * 2, curr_dim, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(curr_dim),
            nn.ReLU(inplace=True))
        # 处理过程
        self.process = nn.Sequential(
            nn.Conv2d(curr_dim * 2, curr_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(curr_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(curr_dim, 3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(curr_dim),
            nn.ReLU(inplace=True))
        # 残差
        self.residual = nn.Sequential(
            nn.Conv2d(6, 3, kernel_size=3, stride=1, padding=1, bias=False), nn.Tanh())
    def forward(self, x, mixed_feature):
        # 正向传播
        # 翻译,连接,准备工作,第一卷积,第二卷积,残差连接, 第一逆卷积, 连接, 第二逆卷积, 连接, process, 连接, 残差
        # Unet做连接是通过 同维度矩阵 拼接来融合特征的
        mixed_feature = self.translator(mixed_feature)
        tmp_tensor = torch.cat((x[:,0,:,:].unsqueeze(1), mixed_feature, x[:,1,:,:].unsqueeze(1), mixed_feature, x[:,2,:,:].unsqueeze(1), mixed_feature), dim = 1)
        tmp_tensor = self.preprocess(tmp_tensor)
        tmp_tensor_first = tmp_tensor
        tmp_tensor = self.firstconv(tmp_tensor)
        tmp_tensor_second = tmp_tensor
        tmp_tensor = self.secondconv(tmp_tensor)
        tmp_tensor = self.residualblock(tmp_tensor)

        tmp_tensor = self.firstconvtrans(tmp_tensor)
        tmp_tensor = torch.cat((tmp_tensor_second, tmp_tensor), dim = 1)
        tmp_tensor = self.secondconvtrans(tmp_tensor)
        tmp_tensor = torch.cat((tmp_tensor_first, tmp_tensor), dim = 1)
        tmp_tensor = self.process(tmp_tensor)
        tmp_tensor = torch.cat((x, tmp_tensor), dim = 1)
        tmp_tensor = self.residual(tmp_tensor)
        return tmp_tensor


# ResidualBlock
# 残差块
class ResidualBlock(nn.Module):
    def __init__(self, dim_in, dim_out, net_mode=None):
        if net_mode == 'p' or (net_mode is None):
            use_affine = True
        elif net_mode == 't':
            use_affine = False
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in,
                      dim_out,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False), nn.InstanceNorm2d(dim_out,
                                                     affine=use_affine),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out,
                      dim_out,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False), nn.InstanceNorm2d(dim_out,
                                                     affine=use_affine))
    def forward(self, x):
        return x + self.main(x)


# Discriminator
# 判别器
# 此外，鉴别器加强了视觉现实主义和良性和对抗性例子之间的类别区分。
# 由benign样本和对抗样本 作为输入 经过判别器 最大化 预测图片是真实数据还是生成器生成的数据 的准确率。
class Discriminator(nn.Module):
    def __init__(self, num_classes, image_size=224, conv_dim=64, repeat_num=5):
        super(Discriminator, self).__init__()
        layers = []
        layers.append(SpectralNorm(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1)))
        layers.append(nn.LeakyReLU(0.01))
        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1)))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2
        kernel_size = int(image_size / (2**repeat_num))
        self.main = nn.Sequential(*layers)
        self.fc = nn.Conv2d(curr_dim, num_classes + 1, kernel_size=kernel_size, bias=False)

    def forward(self, x):
        h = self.main(x)
        out = self.fc(h)
        return out.squeeze()


# GAN Objectives
# 生成对抗网络的目标，GAN损失函数
class GANLoss(nn.Module):
    def __init__(self, gan_mode, target_real_label=0.0, target_fake_label=1.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)
    def get_target_tensor(self, label, target_is_real):
        if target_is_real:
            real_label = self.real_label.expand(label.size(0), 1)
            target_tensor = torch.cat([label, real_label], dim=-1)
        else:
            fake_label = self.fake_label.expand(label.size(0), 1)
            target_tensor = torch.cat([label, fake_label], dim=-1)
        return target_tensor
    def __call__(self, prediction, label, target_is_real):
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(label, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


# Learning Rate Scheduler
# 学习率调度程序.经过一定epoch迭代以后,模型效果不再提升,表示该学习率可能不再适应改模型
# 于是在训练过程中缩小学习率，进而提升模型.
def get_scheduler(optimizer, opt):
    # 线性下降
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count -
                             opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    #     按步下降
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer,
                                        step_size=opt.lr_decay_iters,
                                        gamma=0.1)
    #使用keras中的回调函数,ReduceLROnPlateau
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
                                                   mode='min',
                                                   factor=0.2,
                                                   threshold=0.01,
                                                   patience=5)
    #     余弦函数优化学习率
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer,
                                                   T_max=opt.n_epochs,
                                                   eta_min=0)
    else:
        return NotImplementedError(
            'learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler