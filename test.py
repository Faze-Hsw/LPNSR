#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
超分辨率模型测试脚本

功能：
1. 加载LQ图像，使用训练好的模型生成SR图像
2. 如果有GT图像，计算全参考指标（PSNR, SSIM, LPIPS）
3. 计算无参考指标（NIQE, PI, CLIP-IQA, MUSIQ）
4. 输出评估结果到CSV/JSON文件
"""

import os
import sys
import warnings

# 过滤警告
warnings.filterwarnings("ignore", message=".*A matching Triton is not available.*")
warnings.filterwarnings("ignore", message=".*No module named 'triton'.*")

import argparse
import yaml
import json
import csv
import torch
import numpy as np
import cv2
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from collections import OrderedDict

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

# 导入推理器
from LPNSR.inference import NoisePredictorInference


class MetricsCalculator:
    """
    指标计算器
    
    支持以下指标：
    - 全参考指标: PSNR, SSIM, LPIPS
    - 无参考指标: NIQE, PI, CLIP-IQA, MUSIQ
    """
    
    def __init__(self, config: dict, device: str = 'cuda'):
        """
        初始化指标计算器
        
        Args:
            config: 指标配置
            device: 计算设备
        """
        self.config = config
        self.device = device
        self.use_pyiqa = config.get('use_pyiqa', True)
        
        # 指标参数
        self.crop_border = config.get('crop_border', 4)
        self.test_y_channel = config.get('test_y_channel', True)
        self.lpips_net = config.get('lpips_net', 'alex')
        
        # 初始化指标计算器
        self._init_metrics()
    
    def _init_metrics(self):
        """初始化各种指标计算器"""
        self.metrics_enabled = {}
        self.metric_calculators = {}
        
        # 检查是否可以使用pyiqa库
        self.pyiqa_available = False
        if self.use_pyiqa:
            try:
                import pyiqa
                self.pyiqa_available = True
                print("✓ pyiqa库可用，将使用官方实现")
            except ImportError:
                print("⚠ pyiqa库不可用，将使用内置实现")
                print("  建议安装: pip install pyiqa")
        
        # 初始化全参考指标
        # PSNR
        if self.config.get('calculate_psnr', True):
            self.metrics_enabled['psnr'] = True
            from LPNSR.metrics.psnr import PSNR
            self.metric_calculators['psnr'] = PSNR(
                crop_border=self.crop_border,
                test_y_channel=self.test_y_channel
            )
            print(f"  ✓ PSNR 初始化完成")
        
        # SSIM
        if self.config.get('calculate_ssim', True):
            self.metrics_enabled['ssim'] = True
            from LPNSR.metrics.ssim import SSIM
            self.metric_calculators['ssim'] = SSIM(
                crop_border=self.crop_border,
                test_y_channel=self.test_y_channel
            )
            print(f"  ✓ SSIM 初始化完成")
        
        # LPIPS
        if self.config.get('calculate_lpips', True):
            self.metrics_enabled['lpips'] = True
            if self.pyiqa_available:
                import pyiqa
                self.metric_calculators['lpips'] = pyiqa.create_metric(
                    'lpips', device=self.device, net=self.lpips_net
                )
            else:
                from LPNSR.metrics.lpips import LPIPS
                self.metric_calculators['lpips'] = LPIPS(
                    net=self.lpips_net,
                    use_gpu=(self.device == 'cuda')
                )
            print(f"  ✓ LPIPS 初始化完成 (net={self.lpips_net})")
        
        # 初始化无参考指标
        # NIQE
        if self.config.get('calculate_niqe', True):
            self.metrics_enabled['niqe'] = True
            if self.pyiqa_available:
                import pyiqa
                self.metric_calculators['niqe'] = pyiqa.create_metric('niqe', device=self.device)
            else:
                from LPNSR.metrics.niqe import NIQE
                self.metric_calculators['niqe'] = NIQE(device=self.device)
            print(f"  ✓ NIQE 初始化完成")
        
        # PI
        if self.config.get('calculate_pi', True):
            self.metrics_enabled['pi'] = True
            if self.pyiqa_available:
                try:
                    import pyiqa
                    self.metric_calculators['pi'] = pyiqa.create_metric('pi', device=self.device)
                except Exception as e:
                    print(f"  ⚠ PI (pyiqa) 初始化失败: {e}，使用内置实现")
                    from LPNSR.metrics.pi import PI
                    self.metric_calculators['pi'] = PI(device=self.device)
            else:
                from LPNSR.metrics.pi import PI
                self.metric_calculators['pi'] = PI(device=self.device)
            print(f"  ✓ PI 初始化完成")
        
        # CLIP-IQA
        if self.config.get('calculate_clipiqa', True):
            self.metrics_enabled['clipiqa'] = True
            if self.pyiqa_available:
                try:
                    import pyiqa
                    self.metric_calculators['clipiqa'] = pyiqa.create_metric('clipiqa', device=self.device)
                except Exception as e:
                    print(f"  ⚠ CLIP-IQA (pyiqa) 初始化失败: {e}")
                    self.metrics_enabled['clipiqa'] = False
            else:
                from LPNSR.metrics.clipiqa import CLIPIQA
                self.metric_calculators['clipiqa'] = CLIPIQA(device=self.device)
            if self.metrics_enabled['clipiqa']:
                print(f"  ✓ CLIP-IQA 初始化完成")
        
        # MUSIQ
        if self.config.get('calculate_musiq', True):
            self.metrics_enabled['musiq'] = True
            if self.pyiqa_available:
                try:
                    import pyiqa
                    self.metric_calculators['musiq'] = pyiqa.create_metric('musiq', device=self.device)
                except Exception as e:
                    print(f"  ⚠ MUSIQ (pyiqa) 初始化失败: {e}")
                    self.metrics_enabled['musiq'] = False
            else:
                from LPNSR.metrics.musiq import MUSIQ
                self.metric_calculators['musiq'] = MUSIQ(device=self.device)
            if self.metrics_enabled['musiq']:
                print(f"  ✓ MUSIQ 初始化完成")
    
    def _to_tensor(self, img: np.ndarray) -> torch.Tensor:
        """
        将numpy图像转换为PyTorch张量
        
        Args:
            img: numpy图像 (H, W, C), [0, 255], BGR
            
        Returns:
            张量 (1, C, H, W), [0, 1], RGB
        """
        # BGR -> RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # HWC -> CHW
        img_chw = img_rgb.transpose(2, 0, 1)
        # 归一化到[0, 1]
        img_tensor = torch.from_numpy(img_chw).float() / 255.0
        # 添加batch维度
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        return img_tensor
    
    def calculate_fr_metrics(
        self,
        sr_img: np.ndarray,
        gt_img: np.ndarray
    ) -> dict:
        """
        计算全参考指标
        
        Args:
            sr_img: SR图像 (H, W, C), [0, 255], BGR
            gt_img: GT图像 (H, W, C), [0, 255], BGR
            
        Returns:
            指标字典
        """
        results = {}
        
        # PSNR
        if self.metrics_enabled.get('psnr', False):
            psnr_val = self.metric_calculators['psnr'](sr_img, gt_img)
            results['psnr'] = psnr_val
        
        # SSIM
        if self.metrics_enabled.get('ssim', False):
            ssim_val = self.metric_calculators['ssim'](sr_img, gt_img)
            results['ssim'] = ssim_val
        
        # LPIPS
        if self.metrics_enabled.get('lpips', False):
            sr_tensor = self._to_tensor(sr_img)
            gt_tensor = self._to_tensor(gt_img)
            
            with torch.no_grad():
                if self.pyiqa_available:
                    lpips_val = self.metric_calculators['lpips'](sr_tensor, gt_tensor).item()
                else:
                    lpips_val = self.metric_calculators['lpips'](sr_tensor, gt_tensor)
            results['lpips'] = lpips_val
        
        return results
    
    def calculate_nr_metrics(self, sr_img: np.ndarray) -> dict:
        """
        计算无参考指标
        
        Args:
            sr_img: SR图像 (H, W, C), [0, 255], BGR
            
        Returns:
            指标字典
        """
        results = {}
        sr_tensor = self._to_tensor(sr_img)
        
        # NIQE
        if self.metrics_enabled.get('niqe', False):
            with torch.no_grad():
                if self.pyiqa_available and hasattr(self.metric_calculators['niqe'], '__call__'):
                    niqe_val = self.metric_calculators['niqe'](sr_tensor).item()
                else:
                    niqe_val = self.metric_calculators['niqe'](sr_img)
            results['niqe'] = niqe_val
        
        # PI
        if self.metrics_enabled.get('pi', False):
            with torch.no_grad():
                try:
                    if self.pyiqa_available and hasattr(self.metric_calculators['pi'], '__call__'):
                        pi_val = self.metric_calculators['pi'](sr_tensor).item()
                    else:
                        pi_val = self.metric_calculators['pi'](sr_img)
                    results['pi'] = pi_val
                except Exception as e:
                    print(f"  ⚠ PI计算失败: {e}")
                    results['pi'] = float('nan')
        
        # CLIP-IQA
        if self.metrics_enabled.get('clipiqa', False):
            with torch.no_grad():
                try:
                    if self.pyiqa_available:
                        clipiqa_val = self.metric_calculators['clipiqa'](sr_tensor).item()
                    else:
                        clipiqa_val = self.metric_calculators['clipiqa'](sr_img)
                    results['clipiqa'] = clipiqa_val
                except Exception as e:
                    print(f"  ⚠ CLIP-IQA计算失败: {e}")
                    results['clipiqa'] = float('nan')
        
        # MUSIQ
        if self.metrics_enabled.get('musiq', False):
            with torch.no_grad():
                try:
                    if self.pyiqa_available:
                        musiq_val = self.metric_calculators['musiq'](sr_tensor).item()
                    else:
                        musiq_val = self.metric_calculators['musiq'](sr_img)
                    results['musiq'] = musiq_val
                except Exception as e:
                    print(f"  ⚠ MUSIQ计算失败: {e}")
                    results['musiq'] = float('nan')
        
        return results
    
    def calculate_all(
        self,
        sr_img: np.ndarray,
        gt_img: np.ndarray = None
    ) -> dict:
        """
        计算所有指标
        
        Args:
            sr_img: SR图像
            gt_img: GT图像（可选，如果没有则只计算无参考指标）
            
        Returns:
            所有指标的字典
        """
        results = {}
        
        # 计算全参考指标
        if gt_img is not None:
            fr_results = self.calculate_fr_metrics(sr_img, gt_img)
            results.update(fr_results)
        
        # 计算无参考指标
        nr_results = self.calculate_nr_metrics(sr_img)
        results.update(nr_results)
        
        return results


class SRTester:
    """
    超分辨率测试器
    
    功能：
    1. 加载LQ图像并生成SR图像
    2. 计算各种评估指标
    3. 保存结果
    """
    
    def __init__(self, config_path: str):
        """
        初始化测试器
        
        Args:
            config_path: 测试配置文件路径
        """
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # 设置随机种子
        seed = self.config.get('seed', 12345)
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # 设备
        self.device = self.config.get('device', 'cuda')
        if self.device == 'cuda' and not torch.cuda.is_available():
            print("⚠ CUDA不可用，使用CPU")
            self.device = 'cpu'
        
        # 数据路径
        self.gt_folder = self.config['data'].get('gt_folder', '')
        self.lq_folder = self.config['data']['lq_folder']
        self.output_folder = Path(self.config['data'].get('output_folder', './test_results'))
        
        # 检查是否有GT图像
        self.has_gt = bool(self.gt_folder) and Path(self.gt_folder).exists()
        
        # 输出配置
        self.output_config = self.config.get('output', {})
        self.save_sr_images = self.output_config.get('save_sr_images', True)
        self.save_metrics_csv = self.output_config.get('save_metrics_csv', True)
        self.save_metrics_json = self.output_config.get('save_metrics_json', True)
        self.print_per_image = self.output_config.get('print_per_image', True)
        
        # 初始化推理器
        print("\n" + "=" * 60)
        print("初始化推理器...")
        print("=" * 60)
        inference_config = self.config['inference']['config_path']
        self.inferencer = NoisePredictorInference(inference_config, device=self.device)
        
        # 初始化指标计算器
        print("\n" + "=" * 60)
        print("初始化指标计算器...")
        print("=" * 60)
        self.metrics_calculator = MetricsCalculator(
            self.config['metrics'],
            device=self.device
        )
        
        print("\n" + "=" * 60)
        print("测试器初始化完成")
        print("=" * 60)
        print(f"  - LQ文件夹: {self.lq_folder}")
        print(f"  - GT文件夹: {self.gt_folder if self.has_gt else '无（只计算无参考指标）'}")
        print(f"  - 输出文件夹: {self.output_folder}")
        print(f"  - 设备: {self.device}")
    
    def _get_image_pairs(self) -> list:
        """
        获取图像对列表
        
        Returns:
            [(lq_path, gt_path), ...] 列表，gt_path可能为None
        """
        lq_folder = Path(self.lq_folder)
        
        # 支持的图像格式
        extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.PNG', '*.JPG', '*.JPEG', '*.BMP']
        
        # 获取LQ图像
        lq_paths = []
        for ext in extensions:
            lq_paths.extend(lq_folder.glob(ext))
        lq_paths = sorted(list(set(lq_paths)))
        
        if len(lq_paths) == 0:
            raise ValueError(f"在 {lq_folder} 中未找到图像文件")
        
        # 匹配GT图像
        pairs = []
        if self.has_gt:
            gt_folder = Path(self.gt_folder)
            for lq_path in lq_paths:
                # 尝试不同的匹配方式
                gt_path = None
                
                # 1. 完全相同的文件名
                candidate = gt_folder / lq_path.name
                if candidate.exists():
                    gt_path = candidate
                else:
                    # 2. 去除后缀如 _lq, _lr, x4 等
                    stem = lq_path.stem
                    for suffix in ['_lq', '_lr', '_LQ', '_LR', 'x4', 'x2', '_bicubic','_LR4']:
                        if stem.endswith(suffix):
                            stem = stem[:-len(suffix)]
                            break
                    
                    # 尝试不同扩展名
                    for ext in ['.png', '.jpg', '.jpeg', '.bmp']:
                        candidate = gt_folder / (stem + ext)
                        if candidate.exists():
                            gt_path = candidate
                            break
                        # 尝试加上 _gt, _hr 后缀
                        for gt_suffix in ['_gt', '_hr', '_GT', '_HR']:
                            candidate = gt_folder / (stem + gt_suffix + ext)
                            if candidate.exists():
                                gt_path = candidate
                                break
                
                pairs.append((lq_path, gt_path))
        else:
            pairs = [(lq_path, None) for lq_path in lq_paths]
        
        return pairs
    
    def _process_single_image(self, lq_path: Path) -> np.ndarray:
        """
        处理单张图像
        
        Args:
            lq_path: LQ图像路径
            
        Returns:
            SR图像 (H, W, C), [0, 255], BGR
        """
        # 读取LQ图像
        lq_img = cv2.imread(str(lq_path))
        lq_img_rgb = cv2.cvtColor(lq_img, cv2.COLOR_BGR2RGB)
        lq_img_float = lq_img_rgb.astype(np.float32) / 255.0
        
        # 超分辨率处理
        sr_img_float = self.inferencer.process_single_image(lq_img_float)
        
        # 转换为uint8 BGR
        sr_img = (sr_img_float * 255.0).clip(0, 255).astype(np.uint8)
        sr_img_bgr = cv2.cvtColor(sr_img, cv2.COLOR_RGB2BGR)
        
        return sr_img_bgr
    
    def run(self):
        """
        运行测试
        """
        print("\n" + "=" * 60)
        print("开始测试")
        print("=" * 60)
        
        # 创建输出目录
        self.output_folder.mkdir(parents=True, exist_ok=True)
        sr_output_folder = self.output_folder / 'sr_images'
        if self.save_sr_images:
            sr_output_folder.mkdir(parents=True, exist_ok=True)
        
        # 获取图像对
        pairs = self._get_image_pairs()
        print(f"\n找到 {len(pairs)} 张图像")
        
        if self.has_gt:
            gt_count = sum(1 for _, gt in pairs if gt is not None)
            print(f"  - 匹配到GT图像: {gt_count} 张")
        
        # 存储所有结果
        all_results = []
        
        # 处理每张图像
        for lq_path, gt_path in tqdm(pairs, desc="测试进度"):
            result = OrderedDict()
            result['image_name'] = lq_path.name
            
            # 生成SR图像
            sr_img = self._process_single_image(lq_path)
            
            # 保存SR图像
            if self.save_sr_images:
                sr_save_path = sr_output_folder / f"{lq_path.stem}_sr.png"
                cv2.imwrite(str(sr_save_path), sr_img)
            
            # 加载GT图像
            gt_img = None
            if gt_path is not None and gt_path.exists():
                gt_img = cv2.imread(str(gt_path))
                # 确保GT和SR尺寸一致
                if gt_img.shape[:2] != sr_img.shape[:2]:
                    print(f"  ⚠ 尺寸不匹配: SR={sr_img.shape[:2]}, GT={gt_img.shape[:2]}，跳过全参考指标")
                    gt_img = None
            
            # 计算指标
            metrics = self.metrics_calculator.calculate_all(sr_img, gt_img)
            result.update(metrics)
            
            # 打印单张图像结果
            if self.print_per_image:
                metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items() if not np.isnan(v)])
                print(f"\n  {lq_path.name}: {metrics_str}")
            
            all_results.append(result)
        
        # 计算平均值
        print("\n" + "=" * 60)
        print("测试结果汇总")
        print("=" * 60)
        
        avg_results = OrderedDict()
        metric_keys = [k for k in all_results[0].keys() if k != 'image_name']
        
        for key in metric_keys:
            values = [r[key] for r in all_results if key in r and not np.isnan(r.get(key, float('nan')))]
            if values:
                avg_results[key] = np.mean(values)
                std = np.std(values)
                print(f"  {key.upper():10s}: {avg_results[key]:.4f} ± {std:.4f}")
        
        # 保存结果到CSV
        if self.save_metrics_csv:
            csv_path = self.output_folder / 'metrics.csv'
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
                writer.writeheader()
                writer.writerows(all_results)
                # 添加平均值行
                avg_row = {'image_name': 'AVERAGE'}
                avg_row.update(avg_results)
                writer.writerow(avg_row)
            print(f"\n✓ 指标已保存到: {csv_path}")
        
        # 保存结果到JSON
        if self.save_metrics_json:
            json_path = self.output_folder / 'metrics.json'
            output_data = {
                'timestamp': datetime.now().isoformat(),
                'config': {
                    'lq_folder': str(self.lq_folder),
                    'gt_folder': str(self.gt_folder) if self.has_gt else None,
                    'metrics_config': self.config['metrics']
                },
                'average': {k: float(v) for k, v in avg_results.items()},
                'per_image': [{k: (float(v) if isinstance(v, (int, float, np.floating)) else v) 
                              for k, v in r.items()} for r in all_results]
            }
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            print(f"✓ 指标已保存到: {json_path}")
        
        if self.save_sr_images:
            print(f"✓ SR图像已保存到: {sr_output_folder}")
        
        print("\n" + "=" * 60)
        print("测试完成！")
        print("=" * 60)
        
        return avg_results


def get_parser():
    """获取命令行参数解析器"""
    parser = argparse.ArgumentParser(description="超分辨率模型测试脚本")
    
    parser.add_argument(
        "-c", "--config",
        type=str,
        default="LPNSR/configs/test_config.yaml",
        help="测试配置文件路径"
    )
    parser.add_argument(
        "--lq_folder",
        type=str,
        default=None,
        help="LQ图像文件夹路径（覆盖配置文件）"
    )
    parser.add_argument(
        "--gt_folder",
        type=str,
        default=None,
        help="GT图像文件夹路径（覆盖配置文件）"
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default=None,
        help="输出文件夹路径（覆盖配置文件）"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu"],
        help="计算设备（覆盖配置文件）"
    )
    parser.add_argument(
        "--no_save_sr",
        action="store_true",
        help="不保存SR图像"
    )
    
    return parser


def main():
    """主函数"""
    parser = get_parser()
    args = parser.parse_args()
    
    print("=" * 60)
    print("超分辨率模型测试脚本")
    print("=" * 60)
    
    # 加载配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 命令行参数覆盖配置
    if args.lq_folder:
        config['data']['lq_folder'] = args.lq_folder
    if args.gt_folder:
        config['data']['gt_folder'] = args.gt_folder
    if args.output_folder:
        config['data']['output_folder'] = args.output_folder
    if args.device:
        config['device'] = args.device
    if args.no_save_sr:
        config['output']['save_sr_images'] = False
    
    # 检查必要参数
    if not config['data'].get('lq_folder'):
        raise ValueError("必须指定LQ图像文件夹路径（通过配置文件或--lq_folder参数）")
    
    # 临时保存修改后的配置
    temp_config_path = Path(args.config).parent / 'test_config_temp.yaml'
    with open(temp_config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True)
    
    try:
        # 创建测试器并运行
        tester = SRTester(str(temp_config_path))
        results = tester.run()
    finally:
        # 清理临时文件
        if temp_config_path.exists():
            temp_config_path.unlink()
    
    return results


if __name__ == '__main__':
    main()
