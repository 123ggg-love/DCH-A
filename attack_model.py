import os
import numpy as np
import scipy.io as scio
from torchvision import transforms
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import PrototypeNet, Generator, Discriminator, GANLoss, get_scheduler
from utils import set_input_images, CalcSim, log_trick, CalcMap, mkdir_p
from attacked_model import Attacked_Model

# 攻击模型
# 定义类: 本文的攻击方法DCHTA
# 其中包含初始化__init__,
# _build_model
# _save_setting
# sample
# set_requires_grad
# update_learning_rate

# save_prototypenet
# save_generator
# load_generator
# load_prototypenet

# train_prototypenet
# test_prototypenet
# train
# test
# transfer_attack
#  .
class DCHTA(nn.Module):
    def __init__(self, args, Dcfg):
        super(DCHTA, self).__init__()
        # 二值码位数
        self.bit = args.bit
        # 类别标签数目
        self.num_classes = Dcfg.num_label
        # 文本维度
        self.dim_text = Dcfg.tag_dim
        #
        self.batch_size = args.batch_size
        # 模型名称  使用到的数据集，攻击方法 位数
        self.model_name = '{}_{}_{}'.format(args.dataset, args.attacked_method, args.bit)

        self.args = args
        # 建立模型
        self._build_model(args, Dcfg)
        self._save_setting(args)
        # 判断是基于迁移的攻击,在原模型上生成对抗示例,然后将其转移到目标模型,这种方法的攻击不能得到很高的成功率
        if self.args.transfer_attack:
            self.transfer_bit = args.transfer_bit
            self.transfer_model = Attacked_Model(args.transfer_attacked_method, args.dataset, args.transfer_bit, args.attacked_models_path, args.dataset_path)
            self.transfer_model.eval()
    # 构建攻击模型,包含原型网络(融合模块和语义翻译网络),生成器,判别器,损失,被攻击模型
    def _build_model(self, args, Dcfg):
        pretrain_model = scio.loadmat(Dcfg.vgg_path)
        # 原型网络,即progressive融合模块,提取细粒度信息
        self.prototypenet = nn.DataParallel(PrototypeNet(self.dim_text, self.bit, self.num_classes)).cuda()
        # 生成器 使得良性样本和嵌入样本作为输入来生成视觉上和benign样本一致的对抗样例
        self.generator = nn.DataParallel(Generator()).cuda()
        # 判别器 使得benign样本和对抗样本视觉上一致,通过将样本分给特定的类别来增强目标语义
        self.discriminator = nn.DataParallel(Discriminator(self.num_classes)).cuda()
        # GAN的目标
        self.criterionGAN = GANLoss('lsgan').cuda()
        # 被攻击模型
        self.attacked_model = Attacked_Model(args.attacked_method, args.dataset, args.bit, args.attacked_models_path, args.dataset_path)

        self.attacked_model.eval()

    # 输出结果的保存
    def _save_setting(self, args):
        # 输出路径
        self.output_dir = os.path.join(args.output_path, args.output_dir)
        # 模型路径
        self.model_dir = os.path.join(self.output_dir, 'Model')
        #
        self.image_dir = os.path.join(self.output_dir, 'Image')
        # 创建文件夹
        mkdir_p(self.model_dir)
        mkdir_p(self.image_dir)
    # ？
    def sample(self, image, sample_dir, name):
        if not os.path.exists(sample_dir):
            os.makedirs(sample_dir)
        image = image.cpu().detach()[0]
        image = transforms.ToPILImage()(image)
        image.convert(mode='RGB').save(os.path.join(sample_dir, name + '.png'), quality=100)

    # 是否可求梯度
    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    # 更新学习率
    def update_learning_rate(self):
        for scheduler in self.schedulers:
            if self.args.lr_policy == 'plateau':
                scheduler.step(0)
            else:
                scheduler.step()
        self.args.lr = self.optimizers[0].param_groups[0]['lr']

    # 保存原型网络
    def save_prototypenet(self):
        torch.save(self.prototypenet.module.state_dict(),
            os.path.join(self.model_dir, 'prototypenet_{}.pth'.format(self.model_name)))
    # 保存生成器网络
    def save_generator(self):
        torch.save(self.generator.module.state_dict(),
            os.path.join(self.model_dir, 'generator_{}.pth'.format(self.model_name)))
    # 下载生成器
    def load_generator(self):
        self.generator.module.load_state_dict(torch.load(os.path.join(self.model_dir, 'generator_{}.pth'.format(self.model_name))))
        self.generator.eval()
    # 下载 progressive融合模块的原型网络
    def load_prototypenet(self):
        self.prototypenet.module.load_state_dict(torch.load(os.path.join(self.model_dir, 'prototypenet_{}.pth'.format(self.model_name))))
        self.prototypenet.eval()

    # 跨模态原型学习
    # 训练原型网络
    def train_prototypenet(self, train_images, train_texts, train_labels):
        # 训练样本的总数目
        num_train = train_labels.size(0)
        # 采用Adam优化器
        optimizer_a = torch.optim.Adam(self.prototypenet.parameters(), lr=self.args.lr, betas=(0.5, 0.999))
        # epochs = 50
        epochs = 10
        # batch_size = 64
        batch_size = 8
        # 这些样本一共分成steps块
        steps = num_train // batch_size + 1

        lr_steps = epochs * steps
        # 多步学习率下降,参数包括:optimizer_a包装优化器,milestones:epoch索引列表(增加),gamma:学习率衰减的乘法因子默认0.1,
        scheduler_a = torch.optim.lr_scheduler.MultiStepLR(optimizer_a, milestones=[lr_steps / 2, lr_steps * 3 / 4], gamma=0.1)
        # MSE损失函数
        criterion_l2 = torch.nn.MSELoss()

        # Depends on the attacked model
        # 被攻击模型根据训练的图像生成图像哈希码
        B = self.attacked_model.generate_image_hashcode(train_images).cuda()
        # B = self.attacked_model.generate_text_hashcode(train_texts).cuda()
        for epoch in range(epochs):
            # 对得到的训练数据进行随机排列序列
            index = np.random.permutation(num_train)

            for i in range(steps):
                # 对于这一个小batch中，定义
                end_index = min((i+1)*batch_size, num_train)
                num_index = end_index - i*batch_size
                ind = index[i*batch_size : end_index]
                # 创建训练 文本和标签 的 变量，该变量可以是任意形状和类型的张量
                batch_text = Variable(train_texts[ind]).type(torch.float).cuda()
                batch_label = Variable(train_labels[ind]).type(torch.float).cuda()
                # 把所有Variable的grad成员数值变为0 变量会被优化器的相关函数更新
                optimizer_a.zero_grad()

                _, mixed_h, mixed_l = self.prototypenet(batch_label, batch_text)
                # 计算sim 目标标签和示例的相似度
                S = CalcSim(batch_label.cpu(), train_labels.type(torch.float))

                theta_m = mixed_h.mm(Variable(B).t()) / 2   #/2转换成汉明距离
                # 创建变量
                logloss_m = - ((Variable(S.cuda()) * theta_m - log_trick(theta_m)).sum() / (num_train * num_index))
                regterm_m = (torch.sign(mixed_h) - mixed_h).pow(2).sum() / num_index
                # l2范数
                classifer_m = criterion_l2(mixed_l, batch_label)
                loss = classifer_m + 5 * logloss_m + 1e-3 * regterm_m
                # 反向传播
                loss.backward()
                # 优化
                optimizer_a.step()
                if i % self.args.print_freq == 0:
                    print('epoch: {:2d}, step: {:3d}, lr: {:.5f}, l_m:{:.5f}, r_m: {:.5f}, c_m: {:.7f}'
                        .format(epoch, i, scheduler_a.get_last_lr()[0], logloss_m, regterm_m, classifer_m))
                scheduler_a.step()
        #   保存原型网络
        self.save_prototypenet()

    def test_prototypenet(self, test_texts, test_labels, database_images, database_texts, database_labels):
        # 加载原型网络
        self.load_prototypenet()
        # 测试集的数目
        num_test = test_labels.size(0)
        # 初始化测试的qB
        qB = torch.zeros([num_test, self.bit])
        for i in range(num_test):
            # 原型网络的测试的标签和文本
            _, mixed_h, __ = self.prototypenet(test_labels[i].cuda().float().unsqueeze(0), test_texts[i].cuda().float().unsqueeze(0))
            # 对实值码进行sign操作，得到二进制码
            qB[i, :] = torch.sign(mixed_h.cpu().data)[0]
        #     对于数据库中的图像运用 攻击模型 生成图像哈希码 IdB
        IdB = self.attacked_model.generate_image_hashcode(database_images)
        # 运用攻击模型生成文本哈希码
        TdB = self.attacked_model.generate_text_hashcode(database_texts)
        # 计算map
        c2i_map = CalcMap(qB, IdB, test_labels, database_labels, 50)

        c2t_map = CalcMap(qB, TdB, test_labels, database_labels, 50)
        # 输出测试集原型网络的map
        print('C2I_MAP: %3.5f, C2T_MAP: %3.5f' % (c2i_map, c2t_map))

    # Overrides method in Module覆盖module中的方法

    def train(self, train_images, train_texts, train_labels, database_images, database_texts, database_labels, test_texts, test_labels):
        # Stage I: Prototype Learning
        self.train_prototypenet(train_images, train_texts, train_labels)
        self.test_prototypenet(test_texts, test_labels, database_images, database_texts, database_labels)

        # Stage II: Adversarial Generation
        # 对抗生成阶段
        # 生成器和判别器的优化器
        optimizer_g = torch.optim.Adam(self.generator.parameters(), lr=self.args.lr, betas=(0.5, 0.999))
        optimizer_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.args.lr, betas=(0.5, 0.999))
        # 两个优化器
        self.optimizers = [optimizer_g, optimizer_d]
        # 调度程序, scheduler主要是为了在训练中以不同的策略来调整学习率
        self.schedulers = [get_scheduler(opt, self.args) for opt in self.optimizers]
        # 训练总数
        num_train = train_labels.size(0)
        # 一个批量的大小
        batch_size = self.batch_size
        # 总epoch
        total_epochs = self.args.n_epochs + self.args.n_epochs_decay + 1
        # l2范数
        criterion_l2 = torch.nn.MSELoss()
        # 输入训练文本，使得被攻击模型生成文本特征
        B = self.attacked_model.generate_text_feature(train_texts)
        B = B.cuda()

        for epoch in range(self.args.epoch_count, total_epochs):
            # 训练，学习率
            print('\nTrain epoch: {}, learning rate: {:.7f}'.format(epoch, self.args.lr))
            # 随机排列序列
            index = np.random.permutation(num_train)
            for i in range(num_train // batch_size + 1):
                end_index = min((i+1)*batch_size, num_train)
                num_index = end_index - i*batch_size
                ind = index[i*batch_size : end_index]
                # 获取训练集中标签图像文本
                batch_label = Variable(train_labels[ind]).type(torch.float).cuda()
                batch_text = Variable(train_texts[ind]).type(torch.float).cuda()
                batch_image = Variable(train_images[ind]).type(torch.float).cuda()
                batch_image = set_input_images(batch_image/255)
                select_index = np.random.choice(range(train_labels.size(0)), size=num_index)
                # 目标样本的标签和文本
                batch_target_label = train_labels.index_select(0, torch.from_numpy(select_index)).type(torch.float).cuda()
                batch_target_text = train_texts.index_select(0, torch.from_numpy(select_index)).type(torch.float).cuda()
                # 是由获得的 目标样本的标签和文本 通过prototypenet原型网络获得 标签特征 和目标哈希码
                label_feature, target_hashcode, _ = self.prototypenet(batch_target_label, batch_target_text)

                # 生成伪图
                batch_fake_image = self.generator(batch_image, label_feature.detach())
                # update D
                # 更新判别器
                if i % 3 == 0:
                    self.set_requires_grad(self.discriminator, True)
                    optimizer_d.zero_grad()
                    batch_image_d = self.discriminator(batch_image)
                    batch_fake_image_d = self.discriminator(batch_fake_image.detach())
                    real_d_loss = self.criterionGAN(batch_image_d, batch_label, True)
                    fake_d_loss = self.criterionGAN(batch_fake_image_d, batch_target_label, False)
                    d_loss = (real_d_loss + fake_d_loss) / 2
                    d_loss.backward()
                    optimizer_d.step()
                # update G
                # 更新生成器
                self.set_requires_grad(self.discriminator, False)
                optimizer_g.zero_grad()
                batch_fake_image_m = (batch_fake_image + 1) / 2 * 255
                predicted_target_hash = self.attacked_model.image_model(batch_fake_image_m)
                # 预测的目标哈希和目标哈希之间的log损失
                logloss = - torch.mean(predicted_target_hash * target_hashcode) + 1
                batch_fake_image_d = self.discriminator(batch_fake_image)
                # GAN损失
                fake_g_loss = self.criterionGAN(batch_fake_image_d, batch_target_label, True)
                # 重构损失
                reconstruction_loss_l = criterion_l2(batch_fake_image, batch_image)
                # backpropagation
                g_loss = 5 * logloss + 1 * fake_g_loss + 150 * reconstruction_loss_l
                g_loss.backward()
                optimizer_g.step()
                if i % self.args.sample_freq == 0:
                    self.sample((batch_fake_image + 1) / 2, '{}/'.format(self.image_dir), str(epoch) + '_' + str(i) + '_fake')
                    self.sample((batch_image + 1) / 2, '{}/'.format(self.image_dir), str(epoch) + '_' + str(i) + '_real')
                if i % self.args.print_freq == 0:
                    print('step: {:3d} d_loss: {:.3f} g_loss: {:.3f} fake_g_loss: {:.3f} logloss: {:.3f} r_loss_l: {:.7f}'
                        .format(i, d_loss, g_loss, fake_g_loss, logloss, reconstruction_loss_l))
            #         更新学习率
            self.update_learning_rate()
        self.save_generator()

    def test(self, database_images, database_texts, database_labels, test_images, test_texts, test_labels):
        self.load_prototypenet()
        self.load_generator()
        num_test = test_labels.size(0)
        qB = torch.zeros([num_test, self.bit])
        perceptibility = 0
        select_index = np.random.choice(range(database_labels.size(0)), size = test_labels.size(0))
        # 目标标签和文本
        target_labels = database_labels.type(torch.float).index_select(0, torch.from_numpy(select_index)).cuda()
        target_texts = database_texts.type(torch.float).index_select(0, torch.from_numpy(select_index)).cuda()
        print('start generate target images...')
        for i in range(num_test):
            label_feature, _, __ =  self.prototypenet(target_labels[i].unsqueeze(0), target_texts[i].unsqueeze(0))
            original_image = set_input_images(test_images[i].type(torch.float).cuda()/255)
            fake_image = self.generator(original_image.unsqueeze(0), label_feature)
            fake_image = (fake_image + 1) / 2
            original_image = (original_image + 1) / 2
            target_image = 255 * fake_image
            target_hashcode = self.attacked_model.generate_image_hashcode(target_image)
            qB[i, :] = torch.sign(target_hashcode.cpu().data)
            perceptibility += F.mse_loss(original_image, fake_image[0]).data
        print('generate target images end!')

        TdB = self.attacked_model.generate_text_hashcode(database_texts)
        IdB = self.attacked_model.generate_image_hashcode(database_images)
        print('perceptibility: {:.7f}'.format(torch.sqrt(perceptibility/num_test)))

        I2T_t_map = CalcMap(qB, TdB, target_labels.cpu(), database_labels.float(), 50)
        I2I_t_map = CalcMap(qB, IdB, target_labels.cpu(), database_labels.float(), 50)
        I2T_map = CalcMap(qB, TdB, test_labels.float().cpu(), database_labels.float(), 50)
        I2I_map = CalcMap(qB, IdB, test_labels.float().cpu(), database_labels.float(), 50)
        print('I2T_tMAP: %3.5f' % (I2T_t_map))
        print('I2I_tMAP: %3.5f' % (I2I_t_map))
        print('I2T_MAP: %3.5f' % (I2T_map))
        print('I2I_MAP: %3.5f' % (I2I_map))

    def transfer_attack(self, database_images, database_texts, database_labels, test_images, test_texts, test_labels):
        self.load_prototypenet()
        self.load_generator()
        num_test = test_labels.size(0)
        qB = torch.zeros([num_test, self.transfer_bit])
        perceptibility = 0
        select_index = np.random.choice(range(database_labels.size(0)), size = test_labels.size(0))
        target_labels = database_labels.type(torch.float).index_select(0, torch.from_numpy(select_index)).cuda()
        target_texts = database_texts.type(torch.float).index_select(0, torch.from_numpy(select_index)).cuda()
        print('start generate target images...')
        for i in range(num_test):
            label_feature, _, __ =  self.prototypenet(target_labels[i].unsqueeze(0), target_texts[i].unsqueeze(0))
            original_image = set_input_images(test_images[i].type(torch.float).cuda()/255)
            fake_image = self.generator(original_image.unsqueeze(0), label_feature)
            fake_image = (fake_image + 1) / 2
            original_image = (original_image + 1) / 2
            target_image = 255 * fake_image
            target_hashcode = self.transfer_model.generate_image_hashcode(target_image)
            qB[i, :] = torch.sign(target_hashcode.cpu().data)
            perceptibility += F.mse_loss(original_image, fake_image[0]).data
        print('generate target images end!')

        TdB = self.transfer_model.generate_text_hashcode(database_texts)
        IdB = self.transfer_model.generate_image_hashcode(database_images)
        print('perceptibility: {:.7f}'.format(torch.sqrt(perceptibility/num_test)))

        I2T_t_map = CalcMap(qB, TdB, target_labels.cpu(), database_labels.float(), 50)
        I2I_t_map = CalcMap(qB, IdB, target_labels.cpu(), database_labels.float(), 50)
        I2T_map = CalcMap(qB, TdB, test_labels.float().cpu(), database_labels.float(), 50)
        I2I_map = CalcMap(qB, IdB, test_labels.float().cpu(), database_labels.float(), 50)
        print('I2T_tMAP: %3.5f' % (I2T_t_map))
        print('I2I_tMAP: %3.5f' % (I2I_t_map))
        print('I2T_MAP: %3.5f' % (I2T_map))
        print('I2I_MAP: %3.5f' % (I2I_map))
