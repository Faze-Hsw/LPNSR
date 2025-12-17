"""
测试脚本：可视化噪声预测器的输出和初始化结果

用于诊断噪声预测器的问题：
1. 可视化预测的噪声分布
2. 可视化 prior_sample 的初始化结果 x_T
3. 可视化 VAE 解码后的图像
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import yaml
import math

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from SR.models.noise_predictor import AdaptiveNoisePredictor


def get_named_eta_schedule(
        schedule_name,
        num_diffusion_timesteps,
        min_noise_level,
        etas_end=0.99,
        kappa=1.0,
        **kwargs
):
    """获取ResShift的eta调度"""
    if schedule_name == 'exponential':
        power = kwargs.get('power', 2.0)
        etas_start = min(min_noise_level / kappa, min_noise_level)
        increaser = math.exp(1 / (num_diffusion_timesteps - 1) * math.log(etas_end / etas_start))
        base = np.ones([num_diffusion_timesteps, ]) * increaser
        power_timestep = np.linspace(0, 1, num_diffusion_timesteps, endpoint=True) ** power
        power_timestep *= (num_diffusion_timesteps - 1)
        sqrt_etas = np.power(base, power_timestep) * etas_start
    else:
        raise ValueError(f"未知的schedule_name: {schedule_name}")

    return sqrt_etas


class NoiseVisualizationTester:
    """噪声预测器可视化测试器"""

    def __init__(self, config_path: str, checkpoint_path: str, device: str = 'cuda'):
        """
        Args:
            config_path: 配置文件路径
            checkpoint_path: 噪声预测器权重路径
            device: 设备
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        # 初始化扩散参数
        self._init_diffusion()

        # 加载模型
        self._load_models(checkpoint_path)

        print(f"[INFO] 测试器初始化完成，设备: {self.device}")
        print(f"[INFO] 扩散参数: num_timesteps={self.num_timesteps}, kappa={self.kappa}")
        print(f"[INFO] sqrt_etas: {self.sqrt_etas.numpy()}")

    def _init_diffusion(self):
        """初始化扩散参数"""
        diffusion_config = self.config['diffusion']

        self.num_timesteps = diffusion_config['num_timesteps']
        self.kappa = diffusion_config['kappa']

        # 计算eta调度
        sqrt_etas = get_named_eta_schedule(
            schedule_name=diffusion_config['eta_schedule'],
            num_diffusion_timesteps=self.num_timesteps,
            min_noise_level=diffusion_config['min_noise_level'],
            etas_end=diffusion_config['etas_end'],
            kappa=self.kappa,
            power=diffusion_config['eta_power']
        )

        self.sqrt_etas = torch.from_numpy(sqrt_etas.astype(np.float32))
        self.etas = self.sqrt_etas ** 2

    def _load_models(self, checkpoint_path: str):
        """加载模型"""
        # 加载VAE
        from SR.ldm.models.autoencoder import VQModelTorch
        vae_path = self.config['model']['vae_path']

        # VAE模型结构参数（必须与预训练权重一致）
        ddconfig = {
            'double_z': False,
            'z_channels': 3,
            'resolution': 256,
            'in_channels': 3,
            'out_ch': 3,
            'ch': 128,
            'ch_mult': [1, 2, 4],
            'num_res_blocks': 2,
            'attn_resolutions': [],
            'dropout': 0.0,
            'padding_mode': 'zeros',
        }

        self.vae = VQModelTorch(
            ddconfig=ddconfig,
            n_embed=8192,
            embed_dim=3,
            rank=8,
            lora_alpha=1.0,
            lora_tune_decoder=False,
        )

        # 加载预训练权重
        vae_ckpt = torch.load(vae_path, map_location='cpu')
        if 'state_dict' in vae_ckpt:
            state_dict = vae_ckpt['state_dict']
        else:
            state_dict = vae_ckpt

        # 去除前缀
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key
            if key.startswith('module._orig_mod.'):
                new_key = key.replace('module._orig_mod.', '')
            elif key.startswith('module.'):
                new_key = key.replace('module.', '')
            new_state_dict[new_key] = value

        self.vae.load_state_dict(new_state_dict, strict=False)
        self.vae.to(self.device)
        self.vae.eval()
        print(f"[INFO] VAE加载完成: {vae_path}")

        # 加载噪声预测器
        np_config = self.config['noise_predictor']
        self.noise_predictor = AdaptiveNoisePredictor(
            latent_channels=np_config['latent_channels'],
            model_channels=np_config['model_channels'],
            channel_mult=np_config['channel_mult'],
            num_res_blocks=np_config['num_res_blocks'],
            attention_levels=np_config['attention_levels'],
            num_heads=np_config['num_heads'],
            use_cross_attention=np_config['use_cross_attention'],
            use_frequency_aware=np_config['use_frequency_aware'],
            use_xformers=np_config.get('use_xformers', True),
            double_z=True  # 输出分布
        )

        # 加载权重
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if 'noise_predictor_state_dict' in checkpoint:
            self.noise_predictor.load_state_dict(checkpoint['noise_predictor_state_dict'])
        elif 'model_state_dict' in checkpoint:
            self.noise_predictor.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.noise_predictor.load_state_dict(checkpoint)

        self.noise_predictor.to(self.device)
        self.noise_predictor.eval()
        print(f"[INFO] 噪声预测器加载完成: {checkpoint_path}")

    def load_image(self, image_path: str, target_size: int = 256) -> torch.Tensor:
        """加载并预处理图像"""
        img = Image.open(image_path).convert('RGB')

        # Resize
        img = img.resize((target_size, target_size), Image.LANCZOS)

        # 转换为tensor [0, 1] -> [-1, 1]
        img_np = np.array(img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
        img_tensor = img_tensor * 2 - 1  # [-1, 1]

        return img_tensor.to(self.device)

    def encode_image(self, img: torch.Tensor) -> torch.Tensor:
        """使用VAE编码图像到潜在空间"""
        with torch.no_grad():
            z = self.vae.encode(img)
        return z

    def decode_latent(self, z: torch.Tensor) -> torch.Tensor:
        """使用VAE解码潜在表示到图像"""
        with torch.no_grad():
            img = self.vae.decode(z)
        return img

    def prior_sample(self, z_y: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        计算 prior_sample 初始化

        公式: x_T = y + κ · √η_T · noise

        Args:
            z_y: LR图像的潜在表示 [B, C, H, W]
            noise: 预测的噪声 [B, C, H, W]

        Returns:
            x_T: 初始化的潜在表示 [B, C, H, W]
        """
        t = self.num_timesteps - 1  # 最后一个时间步
        coef = self.kappa * self.sqrt_etas[t].item()
        x_T = z_y + coef * noise
        return x_T

    @torch.no_grad()
    def test_noise_prediction(self, image_path: str, save_dir: str = 'test_outputs'):
        """
        测试噪声预测并可视化结果

        Args:
            image_path: 输入图像路径
            save_dir: 输出保存目录
        """
        os.makedirs(save_dir, exist_ok=True)

        # 1. 加载并编码图像
        print("\n[Step 1] 加载并编码图像...")
        img = self.load_image(image_path)
        z_y = self.encode_image(img)

        print(f"  输入图像形状: {img.shape}")
        print(f"  潜在表示 z_y 形状: {z_y.shape}")
        print(f"  z_y 统计: min={z_y.min():.4f}, max={z_y.max():.4f}, mean={z_y.mean():.4f}, std={z_y.std():.4f}")

        # 2. 使用噪声预测器生成噪声
        print("\n[Step 2] 噪声预测器生成噪声...")
        t = torch.tensor([self.num_timesteps - 1], device=self.device).long()

        # 获取噪声分布
        predicted_noise = self.noise_predictor(z_y, t, sample_posterior=True)

        print(f"  预测噪声形状: {predicted_noise.shape}")
        print(f"  预测噪声统计: min={predicted_noise.min():.4f}, max={predicted_noise.max():.4f}, "
              f"mean={predicted_noise.mean():.4f}, std={predicted_noise.std():.4f}")

        # 3. 生成随机高斯噪声作为对比
        print("\n[Step 3] 生成随机高斯噪声作为对比...")
        random_noise = torch.randn_like(z_y)
        print(f"  随机噪声统计: min={random_noise.min():.4f}, max={random_noise.max():.4f}, "
              f"mean={random_noise.mean():.4f}, std={random_noise.std():.4f}")

        # 4. 计算 prior_sample 初始化
        print("\n[Step 4] 计算 prior_sample 初始化...")
        coef = self.kappa * self.sqrt_etas[-1].item()
        print(f"  κ = {self.kappa}, √η_T = {self.sqrt_etas[-1].item():.4f}, κ·√η_T = {coef:.4f}")

        x_T_predicted = self.prior_sample(z_y, predicted_noise)
        x_T_random = self.prior_sample(z_y, random_noise)

        print(f"  x_T (预测噪声) 统计: min={x_T_predicted.min():.4f}, max={x_T_predicted.max():.4f}, "
              f"mean={x_T_predicted.mean():.4f}, std={x_T_predicted.std():.4f}")
        print(f"  x_T (随机噪声) 统计: min={x_T_random.min():.4f}, max={x_T_random.max():.4f}, "
              f"mean={x_T_random.mean():.4f}, std={x_T_random.std():.4f}")

        # 5. VAE解码
        print("\n[Step 5] VAE解码...")
        decoded_z_y = self.decode_latent(z_y)
        decoded_x_T_predicted = self.decode_latent(x_T_predicted)
        decoded_x_T_random = self.decode_latent(x_T_random)

        # 6. 可视化
        print("\n[Step 6] 生成可视化...")
        self._visualize_results(
            img=img,
            z_y=z_y,
            predicted_noise=predicted_noise,
            random_noise=random_noise,
            x_T_predicted=x_T_predicted,
            x_T_random=x_T_random,
            decoded_z_y=decoded_z_y,
            decoded_x_T_predicted=decoded_x_T_predicted,
            decoded_x_T_random=decoded_x_T_random,
            save_dir=save_dir
        )

        print(f"\n[完成] 结果已保存到 {save_dir}/")

    def _visualize_results(self, img, z_y, predicted_noise, random_noise,
                           x_T_predicted, x_T_random, decoded_z_y,
                           decoded_x_T_predicted, decoded_x_T_random, save_dir):
        """可视化所有结果"""

        def tensor_to_image(t):
            """将tensor转换为可显示的numpy图像"""
            t = t.squeeze(0).cpu()
            if t.shape[0] == 3:  # RGB图像
                t = (t + 1) / 2  # [-1, 1] -> [0, 1]
                t = t.clamp(0, 1)
                return t.permute(1, 2, 0).numpy()
            else:
                return t.numpy()

        def latent_to_vis(z):
            """将潜在空间tensor可视化"""
            z = z.squeeze(0).cpu()
            # 取前3个通道或全部通道的均值
            if z.shape[0] >= 3:
                z_vis = z[:3]  # 取前3通道
            else:
                z_vis = z.mean(dim=0, keepdim=True).expand(3, -1, -1)
            # 归一化到 [0, 1]
            z_min, z_max = z_vis.min(), z_vis.max()
            z_vis = (z_vis - z_min) / (z_max - z_min + 1e-8)
            return z_vis.permute(1, 2, 0).numpy()

        # 创建大图
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))

        # 第一行：输入和潜在表示
        axes[0, 0].imshow(tensor_to_image(img))
        axes[0, 0].set_title('Input Image')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(latent_to_vis(z_y))
        axes[0, 1].set_title(f'z_y (VAE latent)\nmin={z_y.min():.2f}, max={z_y.max():.2f}')
        axes[0, 1].axis('off')

        axes[0, 2].imshow(tensor_to_image(decoded_z_y))
        axes[0, 2].set_title('Decoded z_y')
        axes[0, 2].axis('off')

        # 噪声直方图
        axes[0, 3].hist(predicted_noise.cpu().flatten().numpy(), bins=100, alpha=0.7, label='Predicted', density=True)
        axes[0, 3].hist(random_noise.cpu().flatten().numpy(), bins=100, alpha=0.7, label='Random N(0,1)', density=True)
        axes[0, 3].axvline(x=-3, color='r', linestyle='--', alpha=0.5)
        axes[0, 3].axvline(x=3, color='r', linestyle='--', alpha=0.5)
        axes[0, 3].set_title('Noise Distribution')
        axes[0, 3].legend()
        axes[0, 3].set_xlim(-15, 15)

        # 第二行：预测噪声 vs 随机噪声
        axes[1, 0].imshow(latent_to_vis(predicted_noise))
        axes[1, 0].set_title(f'Predicted Noise\nmin={predicted_noise.min():.2f}, max={predicted_noise.max():.2f}')
        axes[1, 0].axis('off')

        axes[1, 1].imshow(latent_to_vis(random_noise))
        axes[1, 1].set_title(f'Random Noise\nmin={random_noise.min():.2f}, max={random_noise.max():.2f}')
        axes[1, 1].axis('off')

        # 噪声差异
        noise_diff = (predicted_noise - random_noise).abs()
        axes[1, 2].imshow(latent_to_vis(noise_diff))
        axes[1, 2].set_title(f'|Predicted - Random|\nmean={noise_diff.mean():.2f}')
        axes[1, 2].axis('off')

        # 预测噪声各通道直方图
        pred_np = predicted_noise.squeeze(0).cpu().numpy()
        for c in range(min(3, pred_np.shape[0])):
            axes[1, 3].hist(pred_np[c].flatten(), bins=50, alpha=0.5, label=f'Ch{c}', density=True)
        axes[1, 3].set_title('Predicted Noise by Channel')
        axes[1, 3].legend()

        # 第三行：x_T 初始化结果
        coef = self.kappa * self.sqrt_etas[-1].item()

        axes[2, 0].imshow(latent_to_vis(x_T_predicted))
        axes[2, 0].set_title(f'x_T (Predicted)\nmin={x_T_predicted.min():.2f}, max={x_T_predicted.max():.2f}')
        axes[2, 0].axis('off')

        axes[2, 1].imshow(latent_to_vis(x_T_random))
        axes[2, 1].set_title(f'x_T (Random)\nmin={x_T_random.min():.2f}, max={x_T_random.max():.2f}')
        axes[2, 1].axis('off')

        axes[2, 2].imshow(tensor_to_image(decoded_x_T_predicted))
        axes[2, 2].set_title(f'Decoded x_T (Predicted)\nκ·√η_T={coef:.3f}')
        axes[2, 2].axis('off')

        axes[2, 3].imshow(tensor_to_image(decoded_x_T_random))
        axes[2, 3].set_title('Decoded x_T (Random)')
        axes[2, 3].axis('off')

        plt.suptitle('Noise Predictor Visualization\n'
                     f'Formula: x_T = z_y + κ·√η_T·noise, where κ={self.kappa}, √η_T={self.sqrt_etas[-1]:.4f}',
                     fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'noise_visualization.png'), dpi=150, bbox_inches='tight')
        plt.close()

        # 保存单独的统计信息
        stats = {
            'z_y': {
                'min': z_y.min().item(),
                'max': z_y.max().item(),
                'mean': z_y.mean().item(),
                'std': z_y.std().item()
            },
            'predicted_noise': {
                'min': predicted_noise.min().item(),
                'max': predicted_noise.max().item(),
                'mean': predicted_noise.mean().item(),
                'std': predicted_noise.std().item()
            },
            'random_noise': {
                'min': random_noise.min().item(),
                'max': random_noise.max().item(),
                'mean': random_noise.mean().item(),
                'std': random_noise.std().item()
            },
            'x_T_predicted': {
                'min': x_T_predicted.min().item(),
                'max': x_T_predicted.max().item(),
                'mean': x_T_predicted.mean().item(),
                'std': x_T_predicted.std().item()
            },
            'x_T_random': {
                'min': x_T_random.min().item(),
                'max': x_T_random.max().item(),
                'mean': x_T_random.mean().item(),
                'std': x_T_random.std().item()
            },
            'diffusion_params': {
                'kappa': self.kappa,
                'sqrt_eta_T': self.sqrt_etas[-1].item(),
                'coef': coef
            }
        }

        # 保存统计信息
        stats_path = os.path.join(save_dir, 'statistics.txt')
        with open(stats_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("噪声预测器诊断统计\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"扩散参数:\n")
            f.write(f"  κ (kappa) = {self.kappa}\n")
            f.write(f"  √η_T = {self.sqrt_etas[-1].item():.6f}\n")
            f.write(f"  κ·√η_T = {coef:.6f}\n\n")

            f.write(f"z_y (LR latent):\n")
            f.write(f"  范围: [{stats['z_y']['min']:.4f}, {stats['z_y']['max']:.4f}]\n")
            f.write(f"  均值: {stats['z_y']['mean']:.4f}, 标准差: {stats['z_y']['std']:.4f}\n\n")

            f.write(f"预测噪声:\n")
            f.write(f"  范围: [{stats['predicted_noise']['min']:.4f}, {stats['predicted_noise']['max']:.4f}]\n")
            f.write(f"  均值: {stats['predicted_noise']['mean']:.4f}, 标准差: {stats['predicted_noise']['std']:.4f}\n")
            f.write(f"  [警告] 理想范围应该接近 N(0,1)，即 [-3, 3] 包含99.7%的值\n\n")

            f.write(f"随机噪声 (N(0,1) 参考):\n")
            f.write(f"  范围: [{stats['random_noise']['min']:.4f}, {stats['random_noise']['max']:.4f}]\n")
            f.write(f"  均值: {stats['random_noise']['mean']:.4f}, 标准差: {stats['random_noise']['std']:.4f}\n\n")

            f.write(f"x_T (预测噪声初始化):\n")
            f.write(f"  范围: [{stats['x_T_predicted']['min']:.4f}, {stats['x_T_predicted']['max']:.4f}]\n")
            f.write(f"  均值: {stats['x_T_predicted']['mean']:.4f}, 标准差: {stats['x_T_predicted']['std']:.4f}\n\n")

            f.write(f"x_T (随机噪声初始化):\n")
            f.write(f"  范围: [{stats['x_T_random']['min']:.4f}, {stats['x_T_random']['max']:.4f}]\n")
            f.write(f"  均值: {stats['x_T_random']['mean']:.4f}, 标准差: {stats['x_T_random']['std']:.4f}\n\n")

            f.write("=" * 60 + "\n")
            f.write("诊断结论:\n")
            f.write("=" * 60 + "\n")

            # 诊断
            pred_range = stats['predicted_noise']['max'] - stats['predicted_noise']['min']
            if pred_range > 10:
                f.write(f"[错误] 预测噪声范围过大 ({pred_range:.2f})，应该接近6 (正常N(0,1)的99.7%范围)\n")
            else:
                f.write(f"[正确] 预测噪声范围正常 ({pred_range:.2f})\n")

            if abs(stats['predicted_noise']['mean']) > 0.5:
                f.write(f"[错误] 预测噪声均值偏离0 ({stats['predicted_noise']['mean']:.4f})，应该接近0\n")
            else:
                f.write(f"[正确] 预测噪声均值正常 ({stats['predicted_noise']['mean']:.4f})\n")

            xT_range = stats['x_T_predicted']['max'] - stats['x_T_predicted']['min']
            if xT_range > 20:
                f.write(f"[错误] x_T范围过大 ({xT_range:.2f})，VAE解码可能失败\n")
            else:
                f.write(f"[正确] x_T范围正常 ({xT_range:.2f})\n")

        print(f"  统计信息已保存到: {stats_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='噪声预测器可视化测试')
    parser.add_argument('--config', type=str,
                        default='SR/configs/inference_noise_predictor.yaml',
                        help='配置文件路径')
    parser.add_argument('--checkpoint', type=str,
                        default='SR/pretrained/best_model.pth',
                        help='噪声预测器权重路径')
    parser.add_argument('--image', type=str, required=True,
                        help='测试图像路径')
    parser.add_argument('--output', type=str, default='test_outputs',
                        help='输出目录')
    parser.add_argument('--device', type=str, default='cuda',
                        help='设备 (cuda/cpu)')

    args = parser.parse_args()

    # 创建测试器
    tester = NoiseVisualizationTester(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=args.device
    )

    # 运行测试
    tester.test_noise_prediction(
        image_path=args.image,
        save_dir=args.output
    )


if __name__ == '__main__':
    main()