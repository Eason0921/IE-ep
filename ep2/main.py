import argparse
import getpass
import imageio
import json
import os
import random
import torch
import util
from siren import Siren
from torchvision import transforms
from torchvision.utils import save_image
from training import Trainer
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("-ld", "--logdir", help="保存日志的路径", default=f"/home/dyj0921/project/ep2/{getpass.getuser()}")
parser.add_argument("-ni", "--num_iters", help="训练迭代次数", type=int, default=5000)#50000
parser.add_argument("-lr", "--learning_rate", help="学习率", type=float, default=2e-4)
parser.add_argument("-se", "--seed", help="随机种子", type=int, default=random.randint(1, int(1e6)))
parser.add_argument("-iid", "--image_id", help="要训练的图像ID", type=int, default=15)
parser.add_argument("-lss", "--layer_size", help="层大小的整数列表", type=int, default=28)  # 默认28
parser.add_argument("-nl", "--num_layers", help="层数", type=int, default=10)  # 默认10
parser.add_argument("-w0", "--w0", help="SIREN模型的w0参数", type=float, default=30.0)
parser.add_argument("-w0i", "--w0_initial", help="SIREN模型第一层的w0参数", type=float, default=30.0)
parser.add_argument("-nb", "--num_blocks", help="数据分割的块数", type=int, default=16)  # 默认4

args = parser.parse_args()

# 设置torch和cuda
dtype = torch.float32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_tensor_type('torch.cuda.FloatTensor' if torch.cuda.is_available() else 'torch.FloatTensor')

# 设置随机种子
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

min_id, max_id = args.image_id, args.image_id

# 注册平均值的字典（包括全精度和半精度）
results = {'fp_bpp': [], 'fp_psnr': []}

# 创建存储实验的目录
if not os.path.exists(args.logdir):
    os.makedirs(args.logdir)

#数据分割
def get_blocks(data, num_blocks):
    # 获取数据的通道数、高度和宽度
    channels, height, width = data.shape
    # 计算每个方向上的块大小
    block_size_h = height // num_blocks
    block_size_w = width // num_blocks
    blocks = []
    # 遍历每个块
    for i in range(num_blocks):
        for j in range(num_blocks):
            # 提取当前块的数据
            block = data[:, i*block_size_h:(i+1)*block_size_h, j*block_size_w:(j+1)*block_size_w]
            blocks.append(block)
    return blocks

# 适应图像
for i in range(min_id, max_id + 1):
    print(f'图像 {i}')

    # 加载图像
    # img = imageio.imread(f"ep2/kodak-dataset/kodim{str(i).zfill(2)}.png")
    img = imageio.imread(f"ep2/kodak-dataset/img_3.png")
    img = transforms.ToTensor()(img).float().to(device, dtype)

    #数据分割
    num_blocks = int(args.num_blocks ** 0.5)
    blocks = get_blocks(img.cpu().numpy(), num_blocks)
    blocks = torch.tensor(blocks)
    reconstructed_blocks = []
    
    #训练每个块
    for i,block in enumerate(blocks):
        # 设置模型
        func_rep = Siren(
            dim_in=2,
            dim_hidden=args.layer_size,
            dim_out=3,
            num_layers=args.num_layers,
            final_activation=torch.nn.Identity(),
            w0_initial=args.w0_initial,
            w0=args.w0
        ).to(device)
        # 设置训练
        trainer = Trainer(func_rep, args, lr=args.learning_rate)
        coordinates, features = util.to_coordinates_and_features(block)
        coordinates, features = coordinates.to(device), features.to(device)
        
        # 在全精度下训练模型
        trainer.train(coordinates, features, num_iters=args.num_iters)
        print(f'最佳训练PSNR: {trainer.best_vals["psnr"]:.2f}')

        # 记录全精度结果（如果训练多张图片，则每张图片对应的最佳PSNR都存储在results中）
        results['fp_psnr'].append(trainer.best_vals['psnr'])

        # 保存最佳模型
        torch.save(trainer.best_model, args.logdir + f'/best_model_{i}.pt')

        # 更新当前模型为最佳模型
        func_rep.load_state_dict(trainer.best_model)

        # 保存全精度图像重建
        with torch.no_grad():
            img_recon = func_rep(coordinates).reshape(block.shape[1], block.shape[2], 3).permute(2, 0, 1)
            reconstructed_blocks.append(img_recon)
        # 保存单个图像的日志
        with open(args.logdir + f'/logs{i}.json', 'w') as f:
            json.dump(trainer.logs, f)
# 创建一个具有原始形状的新图像
merged_img = np.zeros((img.shape[1], img.shape[2], img.shape[0]))

# 获取块的形状
_, block_size_h, block_size_w = blocks[0].shape

for i, block in enumerate(reconstructed_blocks):
    block = block.cpu().numpy()
    block = np.transpose(block, (1, 2, 0))
    row = i // num_blocks
    col = i % num_blocks
    merged_img[row*block_size_h:(row+1)*block_size_h, col*block_size_w:(col+1)*block_size_w, :] = block

# 将合并后的图像转换为范围[0, 255]
merged_img = np.clip(merged_img * 255, 0, 255)

# 四舍五入并转换为uint8类型
merged_img = np.round(merged_img).astype(np.uint8)

# 将合并后的图像保存为PNG文件
imageio.imwrite(f'/home/dyj0921/project/ep2/{getpass.getuser()}/train{int(args.num_iters)}.png', merged_img)

# 打印所有图片的训练结果
print('全部结果:')
print(results)
with open(args.logdir + f'/results.json', 'w') as f:
    json.dump(results, f)

# 计算并保存聚合结果，计算所有图片的平均结果
results_mean = {key: util.mean(results[key]) for key in results}
with open(args.logdir + f'/results_mean.json', 'w') as f:
    json.dump(results_mean, f)

# 打印平均结果
print('Aggregate results:')
print(f'Full precision, bpp: {results_mean["fp_bpp"]:.2f}, psnr: {results_mean["fp_psnr"]:.2f}')
