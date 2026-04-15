import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import models
from skimage.color import rgb2lab
from .Color import white_balance_adaptive  # 导入白平衡函数


def apply_white_balance_tensor(img):
    """
    对输入的 torch 张量进行白平衡处理。
    输入 img: shape (B, C, H, W)，数值范围 [0, 1]。
    输出: 同样 shape 的 torch 张量，数值范围 [0, 1]。
    """
    img_np = img.detach().cpu().numpy()  # (B, C, H, W)
    processed_imgs = []
    for i in range(img_np.shape[0]):
        im = img_np[i]  # (C, H, W)
        im = np.transpose(im, (1, 2, 0))  # (H, W, C)
        im_uint8 = np.clip(im * 255, 0, 255).astype(np.uint8)
        im_wb = white_balance_adaptive(im_uint8)
        im_wb = im_wb.astype(np.float32) / 255.0
        im_wb = np.transpose(im_wb, (2, 0, 1))  # (C, H, W)
        processed_imgs.append(im_wb)
    processed_imgs = np.stack(processed_imgs, axis=0)
    return torch.tensor(processed_imgs, device=img.device)


# 定义VGG19特征提取网络
class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        return [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]


# 定义对比度能量损失
class ContrastEnergyLoss(nn.Module):
    def __init__(self):
        super(ContrastEnergyLoss, self).__init__()
        self.sobel_kernel_x = torch.tensor([[-1, 0, 1],
                                            [-2, 0, 2],
                                            [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.sobel_kernel_y = torch.tensor([[-1, -2, -1],
                                            [0, 0, 0],
                                            [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    def forward(self, img1, img2):

        wb_img1 = apply_white_balance_tensor(img1)
        wb_img2 = apply_white_balance_tensor(img2)

        self.sobel_kernel_x = self.sobel_kernel_x.to(wb_img1.device)
        self.sobel_kernel_y = self.sobel_kernel_y.to(wb_img1.device)

        # 转为灰度图后计算梯度
        img1_gray = torch.mean(wb_img1, dim=1, keepdim=True)
        img2_gray = torch.mean(wb_img2, dim=1, keepdim=True)

        grad_x1 = F.conv2d(img1_gray, self.sobel_kernel_x, padding=1)
        grad_y1 = F.conv2d(img1_gray, self.sobel_kernel_y, padding=1)
        contrast_energy1 = torch.sqrt(grad_x1 ** 2 + grad_y1 ** 2)

        grad_x2 = F.conv2d(img2_gray, self.sobel_kernel_x, padding=1)
        grad_y2 = F.conv2d(img2_gray, self.sobel_kernel_y, padding=1)
        contrast_energy2 = torch.sqrt(grad_x2 ** 2 + grad_y2 ** 2)

        contrast_energy_loss = F.l1_loss(contrast_energy1, contrast_energy2)
        return contrast_energy_loss


# 定义DCD损失（颜色一致性损失）
class DCDCosineLoss(nn.Module):
    def __init__(self):
        super(DCDCosineLoss, self).__init__()

    def rgb_to_lab(self, image):
        image_np = image.detach().cpu().numpy()
        # 确保 image_np 是四维的，添加批次维度如果需要
        if len(image_np.shape) == 3:
            image_np = np.expand_dims(image_np, axis=0)
        # 交换批次和通道维度
        image_np = image_np.transpose((0, 2, 3, 1))
        lab_image = rgb2lab(image_np)
        if lab_image is None:
            raise ValueError("LAB conversion failed.")
        lab_image = torch.from_numpy(lab_image).permute(0, 3, 1, 2).to(image.device)
        return lab_image

    def forward(self, input, target):

        wb_input = apply_white_balance_tensor(input)
        wb_target = apply_white_balance_tensor(target)

        lab_input = self.rgb_to_lab(wb_input)
        lab_target = self.rgb_to_lab(wb_target)

        # 从 LAB 中提取 a 和 b 通道
        a_input = lab_input[:, 1:2, :, :]
        b_input = lab_input[:, 2:3, :, :]
        a_target = lab_target[:, 1:2, :, :]
        b_target = lab_target[:, 2:3, :, :]

        dot_product = a_input * a_target + b_input * b_target
        norm_input = torch.sqrt(a_input ** 2 + b_input ** 2)
        norm_target = torch.sqrt(a_target ** 2 + b_target ** 2)

        cosine_distance = 1 - (dot_product / (norm_input * norm_target + 1e-8))
        dcd_loss = torch.norm(cosine_distance, p='fro')
        return dcd_loss


# 定义最终的ContrastLoss损失函数
class ContrastLoss(nn.Module):
    def __init__(self, ablation=False, lambda_contrast_energy=0.5, lambda_dcd=0.1):
        super(ContrastLoss, self).__init__()
        self.vgg = Vgg19().cuda()
        self.l1 = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]  # VGG 多层权重
        self.ab = ablation
        self.lambda_contrast_energy = lambda_contrast_energy
        self.contrast_energy_loss_fn = ContrastEnergyLoss()
        self.lambda_dcd = lambda_dcd
        self.dcd_loss_fn = DCDCosineLoss()

    def forward(self, a, p, n=None):
        """
        a: 网络输出（去雾/增强后的图像）
        p: 清晰真实图像（ground truth）
        n: 输入雾霾图像（可选，默认为 None）
        """
        a_vgg, p_vgg = self.vgg(a), self.vgg(p)  # 计算 a 和 p 的 VGG 特征
        loss = 0

        for i in range(len(a_vgg)):
            d_ap = self.l1(a_vgg[i], p_vgg[i].detach())  # 计算 a 与 p 的 L1 损失

            # 如果 ablation 为 False，并且提供了 n，则在前 2 层使用 n 进行正则化
            if not self.ab and n is not None and i < 2:
                n_vgg = self.vgg(n)  # 计算 n 的 VGG 特征
                d_an = self.l1(a_vgg[i], n_vgg[i].detach())
                contrastive = d_ap / (d_an + 1e-7)  # 计算对比损失
            else:
                contrastive = d_ap  # 仅计算 a 和 p 之间的损失

            loss += self.weights[i] * contrastive  # 加权求和

        # 计算额外的损失项
        contrast_energy_loss = self.contrast_energy_loss_fn(a, p)  # 对比能量损失
        dcd_loss = self.dcd_loss_fn(a, p)  # 颜色一致性损失（DCD）

        # 组合损失
        total_loss = loss + self.lambda_contrast_energy * contrast_energy_loss + self.lambda_dcd * dcd_loss
        #total_loss = loss + self.lambda_contrast_energy * contrast_energy_loss
        total_loss = loss
        return total_loss
