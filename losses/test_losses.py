"""
损失函数测试脚本

测试所有实现的损失函数：
1. L2Loss
2. FocalFrequencyLoss
3. LPIPSLoss
"""

import torch
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from SR.losses import L2Loss, FocalFrequencyLoss, LPIPSLoss


def test_l2_loss():
    """测试L2损失"""
    print("\n" + "=" * 70)
    print("测试 L2Loss")
    print("=" * 70)
    
    # 创建损失函数
    criterion = L2Loss(reduction='mean', loss_weight=1.0)
    print(f"\n损失函数: {criterion}")
    
    # 测试数据
    pred = torch.randn(4, 3, 64, 64)
    target = torch.randn(4, 3, 64, 64)
    
    # 计算损失
    loss = criterion(pred, target)
    print(f"\n输入形状: pred={pred.shape}, target={target.shape}")
    print(f"损失值: {loss.item():.6f}")
    
    # 测试梯度
    pred.requires_grad = True
    loss = criterion(pred, target)
    loss.backward()
    print(f"梯度形状: {pred.grad.shape}")
    print(f"梯度范围: [{pred.grad.min().item():.6f}, {pred.grad.max().item():.6f}]")
    
    # 测试带权重的损失
    weight = torch.rand(4, 1, 64, 64)
    loss_weighted = criterion(pred, target, weight=weight)
    print(f"\n带权重的损失值: {loss_weighted.item():.6f}")
    
    print("\n✓ L2Loss 测试通过")


def test_focal_frequency_loss():
    """测试频域感知损失"""
    print("\n" + "=" * 70)
    print("测试 FocalFrequencyLoss")
    print("=" * 70)
    
    # 创建损失函数
    criterion = FocalFrequencyLoss(
        loss_weight=1.0,
        alpha=1.0,
        patch_factor=1,
        ave_spectrum=False,
        log_matrix=False
    )
    print(f"\n损失函数: {criterion}")
    
    # 测试数据
    pred = torch.randn(2, 3, 64, 64)
    target = torch.randn(2, 3, 64, 64)
    
    # 计算损失
    loss = criterion(pred, target)
    print(f"\n输入形状: pred={pred.shape}, target={target.shape}")
    print(f"损失值: {loss.item():.6f}")
    
    # 测试梯度
    pred.requires_grad = True
    loss = criterion(pred, target)
    loss.backward()
    print(f"梯度形状: {pred.grad.shape}")
    print(f"梯度范围: [{pred.grad.min().item():.6f}, {pred.grad.max().item():.6f}]")
    
    # 测试分块模式
    print("\n测试分块模式 (patch_factor=2):")
    criterion_patch = FocalFrequencyLoss(
        loss_weight=1.0,
        alpha=1.0,
        patch_factor=2
    )
    pred_patch = torch.randn(2, 3, 64, 64, requires_grad=True)
    target_patch = torch.randn(2, 3, 64, 64)
    loss_patch = criterion_patch(pred_patch, target_patch)
    print(f"分块损失值: {loss_patch.item():.6f}")
    
    # 测试不同配置
    print("\n测试不同配置:")
    configs = [
        {'ave_spectrum': True, 'log_matrix': False},
        {'ave_spectrum': False, 'log_matrix': True},
        {'ave_spectrum': True, 'log_matrix': True},
    ]
    
    for i, config in enumerate(configs):
        criterion_config = FocalFrequencyLoss(loss_weight=1.0, alpha=1.0, **config)
        loss_config = criterion_config(pred, target)
        print(f"  配置 {i+1} {config}: 损失={loss_config.item():.6f}")
    
    print("\n✓ FocalFrequencyLoss 测试通过")


def test_lpips_loss():
    """测试感知损失（LPIPS）"""
    print("\n" + "=" * 70)
    print("测试 LPIPSLoss")
    print("=" * 70)
    
    # 创建损失函数
    criterion = LPIPSLoss(
        loss_weight=1.0,
        net_type='alex',  # 'alex', 'vgg', 'squeeze'
        spatial=False
    )
    print(f"\n损失函数: {criterion}")
    
    # 测试数据
    pred = torch.randn(2, 3, 64, 64)
    target = torch.randn(2, 3, 64, 64)
    
    # 计算损失
    loss = criterion(pred, target)
    print(f"\n输入形状: pred={pred.shape}, target={target.shape}")
    print(f"损失值: {loss.item():.6f}")
    
    # 测试梯度
    pred.requires_grad = True
    loss = criterion(pred, target)
    loss.backward()
    print(f"梯度形状: {pred.grad.shape}")
    print(f"梯度范围: [{pred.grad.min().item():.6f}, {pred.grad.max().item():.6f}]")
    
    # 测试不同网络类型
    print("\n测试不同网络类型:")
    net_types = ['alex', 'vgg', 'squeeze']
    for net_type in net_types:
        try:
            criterion_net = LPIPSLoss(
                loss_weight=1.0,
                net_type=net_type
            )
            loss_net = criterion_net(pred, target)
            print(f"  网络类型 {net_type}: 损失={loss_net.item():.6f}")
        except Exception as e:
            print(f"  网络类型 {net_type}: 错误={str(e)}")
    
    print("\n✓ LPIPSLoss 测试通过")


def test_combined_loss():
    """测试组合损失"""
    print("\n" + "=" * 70)
    print("测试组合损失")
    print("=" * 70)
    
    # 创建所有损失函数
    l2_loss = L2Loss(loss_weight=1.0)
    freq_loss = FocalFrequencyLoss(loss_weight=0.1, alpha=1.0)
    lpips_loss = LPIPSLoss(loss_weight=0.5, net_type='alex')
    
    print("\n损失函数配置:")
    print(f"  L2Loss: weight=1.0")
    print(f"  FocalFrequencyLoss: weight=0.1")
    print(f"  LPIPSLoss: weight=0.5")
    
    # 测试数据
    pred = torch.randn(2, 3, 128, 128, requires_grad=True)
    target = torch.randn(2, 3, 128, 128)
    
    # 计算各个损失
    loss_l2 = l2_loss(pred, target)
    loss_freq = freq_loss(pred, target)
    loss_lpips = lpips_loss(pred, target)
    
    # 总损失
    total_loss = loss_l2 + loss_freq + loss_lpips
    
    print(f"\n输入形状: pred={pred.shape}, target={target.shape}")
    print(f"\n各项损失值:")
    print(f"  L2 Loss: {loss_l2.item():.6f}")
    print(f"  Frequency Loss: {loss_freq.item():.6f}")
    print(f"  LPIPS Loss: {loss_lpips.item():.6f}")
    print(f"  Total Loss: {total_loss.item():.6f}")
    
    # 测试梯度
    total_loss.backward()
    print(f"\n梯度统计:")
    print(f"  梯度形状: {pred.grad.shape}")
    print(f"  梯度范围: [{pred.grad.min().item():.6f}, {pred.grad.max().item():.6f}]")
    print(f"  梯度均值: {pred.grad.mean().item():.6f}")
    print(f"  梯度标准差: {pred.grad.std().item():.6f}")
    
    print("\n✓ 组合损失测试通过")


def test_performance():
    """测试性能"""
    print("\n" + "=" * 70)
    print("性能测试")
    print("=" * 70)
    
    import time
    
    # 创建损失函数
    l2_loss = L2Loss()
    freq_loss = FocalFrequencyLoss(loss_weight=0.1)
    lpips_loss = LPIPSLoss(loss_weight=0.5, net_type='alex')
    
    # 测试不同分辨率
    resolutions = [64, 128, 256]
    num_iterations = 100
    
    print(f"\n测试配置: batch_size=4, channels=3, iterations={num_iterations}")
    print(f"\n{'分辨率':<10} {'L2 Loss':<15} {'Freq Loss':<15} {'LPIPS Loss':<15}")
    print("-" * 60)
    
    for res in resolutions:
        pred = torch.randn(4, 3, res, res)
        target = torch.randn(4, 3, res, res)
        
        # L2 Loss
        start = time.time()
        for _ in range(num_iterations):
            _ = l2_loss(pred, target)
        time_l2 = (time.time() - start) / num_iterations * 1000
        
        # Frequency Loss
        start = time.time()
        for _ in range(num_iterations):
            _ = freq_loss(pred, target)
        time_freq = (time.time() - start) / num_iterations * 1000
        
        # LPIPS Loss
        start = time.time()
        for _ in range(num_iterations):
            _ = lpips_loss(pred, target)
        time_lpips = (time.time() - start) / num_iterations * 1000
        
        print(f"{res}x{res:<6} {time_l2:<15.3f} {time_freq:<15.3f} {time_lpips:<15.3f} ms")
    
    print("\n✓ 性能测试完成")


def main():
    """主函数"""
    print("\n" + "=" * 70)
    print("超分辨率损失函数测试")
    print("=" * 70)
    
    # 设置随机种子
    torch.manual_seed(42)
    
    try:
        # 测试各个损失函数
        test_l2_loss()
        test_focal_frequency_loss()
        test_lpips_loss()
        test_combined_loss()
        test_performance()
        
        print("\n" + "=" * 70)
        print("✓ 所有测试通过！")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
