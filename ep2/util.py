import numpy as np
import torch
import wandb
from torch._C import dtype
from typing import Dict


DTYPE_BIT_SIZE: Dict[dtype, int] = {
    torch.float32: 32,
    torch.float: 32,
    torch.float64: 64,
    torch.double: 64,
    torch.float16: 16,
    torch.half: 16,
    torch.bfloat16: 16,
    torch.complex32: 32,
    torch.complex64: 64,
    torch.complex128: 128,
    torch.cdouble: 128,
    torch.uint8: 8,
    torch.int8: 8,
    torch.int16: 16,
    torch.short: 16,
    torch.int32: 32,
    torch.int: 32,
    torch.int64: 64,
    torch.long: 64,
    torch.bool: 1
}



def to_coordinates_and_features(img):
    """将图像转换为坐标和特征的集合。

    Args:
        img (torch.Tensor): 形状为 (channels, height, width) 的张量。
    """
    # 坐标是一个与图像空间维度形状相同的全为1的张量的非零位置的索引
    coordinates = torch.ones(img.shape[1:]).nonzero(as_tuple=False).float()
    # 将坐标归一化到 [-.5, .5] 范围内
    coordinates = coordinates / (img.shape[1] - 1) - 0.5
    # 转换到 [-1, 1] 范围内
    coordinates *= 2
    # 将图像转换为形状为 (num_points, channels) 的特征张量
    features = img.reshape(img.shape[0], -1).T
    return coordinates, features


def psnr(img1, img2):
    """计算两个图像之间的峰值信噪比（PSNR）。

    Args:
        img1 (torch.Tensor):
        img2 (torch.Tensor):
    """
    return 20. * np.log10(1.) - 10. * (img1 - img2).detach().pow(2).mean().log10().to('cpu').item()


def clamp_image(img):
    """将图像值限制在 [0, 1] 范围内，并转换为无符号整数。

    Args:
        img (torch.Tensor):
    """
    # 值可能超出 [0, 1] 范围，因此对输入进行限制
    img_ = torch.clamp(img, 0., 1.)
    # 像素值在 {0, ..., 255} 范围内，因此将浮点张量四舍五入
    return torch.round(img_ * 255) / 255.


def get_clamped_psnr(img, img_recon):
    """获取真实图像和重建图像之间的PSNR。由于重建图像来自神经网络的输出，确保值在 [0, 1] 范围内且为无符号整数。

    Args:
        img (torch.Tensor): 真实图像。
        img_recon (torch.Tensor): 模型重建的图像。
    """
    return psnr(img, clamp_image(img_recon))


def mean(list_):
    return np.mean(list_)


def init_wandb(args):
    if args.use_wandb:
        # 初始化wandb实验
        wandb.init(
            project=args.wandb_project_name,  # 项目名称
            entity=args.wandb_entity,  # 实体名称，即个人账号
            job_type=args.wandb_job_type,
            config=args,  # 此次训练的参数设置
        )
