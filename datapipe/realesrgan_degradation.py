"""
RealESRGAN Image Degradation Pipeline Implementation
Implements image degradation process based on configuration file parameters, including blur, noise, resizing, and JPEG compression operations
"""

import cv2
import math
import numpy as np
import random
import sys
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as tv_F
from torchvision import transforms
import yaml
from pathlib import Path
from types import ModuleType
from typing import Dict, List, Tuple, Optional
from PIL import Image

# ---- compatibility: basicsr depends on torchvision.transforms.functional_tensor ----
# torchvision >= 0.15 removed this module, need to create a compatibility shim
if 'torchvision.transforms.functional_tensor' not in sys.modules:
    _ft = ModuleType('torchvision.transforms.functional_tensor')
    _ft.rgb_to_grayscale = tv_F.rgb_to_grayscale
    sys.modules['torchvision.transforms.functional_tensor'] = _ft



class RealESRGANDegradation:
    """
    RealESRGAN Degradation Pipeline Class
    Implements two-stage image degradation process for generating low-quality images
    """
    
    def __init__(self, config_path: str):
        """
        Initialize degradation pipeline
        
        Args:
            config_path: Configuration file path
        """
        self.config = self._load_config(config_path)
        self.opts = self.config['opts']
        self.degradation = self.config['degradation']
        
        # Initialize blur kernel range
        self.kernel_range1 = [x for x in range(3, self.opts['blur_kernel_size'], 2)]
        self.kernel_range2 = [x for x in range(3, self.opts['blur_kernel_size2'], 2)]
        
        # Create impulse tensor (for no blur effect)
        self.pulse_tensor = torch.zeros(
            self.opts['blur_kernel_size2'], 
            self.opts['blur_kernel_size2']
        ).float()
        self.pulse_tensor[
            self.opts['blur_kernel_size2']//2, 
            self.opts['blur_kernel_size2']//2
        ] = 1
        
        # JPEG compressor (initialize when needed)
        self.jpeger = None
    
    def _load_config(self, config_path: str) -> Dict:
        """Load YAML configuration file"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    
    def _sigma_matrix2(self, sig_x: float, sig_y: float, theta: float) -> np.ndarray:
        """
        Calculate rotated sigma matrix (2D matrix)
        
        Args:
            sig_x: Standard deviation in x direction
            sig_y: Standard deviation in y direction
            theta: Rotation angle (in radians)
        
        Returns:
            Rotated sigma matrix
        """
        d_matrix = np.array([[sig_x**2, 0], [0, sig_y**2]])
        u_matrix = np.array([
            [np.cos(theta), -np.sin(theta)], 
            [np.sin(theta), np.cos(theta)]
        ])
        return np.dot(u_matrix, np.dot(d_matrix, u_matrix.T))
    
    def _mesh_grid(self, kernel_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate grid coordinates"""
        ax = np.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
        xx, yy = np.meshgrid(ax, ax)
        return xx, yy
    
    def _pdf2(self, sigma_matrix: np.ndarray, grid_x: np.ndarray, grid_y: np.ndarray) -> np.ndarray:
        """Calculate 2D probability density function"""
        inverse_sigma = np.linalg.inv(sigma_matrix)
        kernel = np.exp(-0.5 * (inverse_sigma[0, 0] * (grid_x**2) + 
                                 inverse_sigma[1, 1] * (grid_y**2) + 
                                 2 * inverse_sigma[0, 1] * grid_x * grid_y))
        return kernel
    
    def _bivariate_gaussian(
        self, 
        kernel_size: int, 
        sig_x: float, 
        sig_y: float, 
        theta: float, 
        grid: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        isotropic: bool = True
    ) -> np.ndarray:
        """
        Generate bivariate Gaussian kernel
        
        Args:
            kernel_size: Kernel size
            sig_x: Standard deviation in x direction
            sig_y: Standard deviation in y direction
            theta: Rotation angle
            grid: Grid coordinates
            isotropic: Whether to use isotropic kernel
        
        Returns:
            Normalized Gaussian kernel
        """
        if grid is None:
            grid = self._mesh_grid(kernel_size)
        
        if isotropic:
            sigma_matrix = np.array([[sig_x**2, 0], [0, sig_x**2]])
        else:
            sigma_matrix = self._sigma_matrix2(sig_x, sig_y, theta)
        
        kernel = self._pdf2(sigma_matrix, grid[0], grid[1])
        kernel = kernel / np.sum(kernel)
        return kernel
    
    def _bivariate_generalized_gaussian(
        self,
        kernel_size: int,
        sig_x: float,
        sig_y: float,
        theta: float,
        beta: float,
        grid: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        isotropic: bool = True
    ) -> np.ndarray:
        """Generate generalized Gaussian kernel"""
        if grid is None:
            grid = self._mesh_grid(kernel_size)
        
        if isotropic:
            sigma_matrix = np.array([[sig_x**2, 0], [0, sig_x**2]])
        else:
            sigma_matrix = self._sigma_matrix2(sig_x, sig_y, theta)
        
        inverse_sigma = np.linalg.inv(sigma_matrix)
        kernel = np.exp(-0.5 * np.power(
            inverse_sigma[0, 0] * (grid[0]**2) + 
            inverse_sigma[1, 1] * (grid[1]**2) + 
            2 * inverse_sigma[0, 1] * grid[0] * grid[1], 
            beta
        ))
        kernel = kernel / np.sum(kernel)
        return kernel
    
    def _bivariate_plateau(
        self,
        kernel_size: int,
        sig_x: float,
        sig_y: float,
        theta: float,
        beta: float,
        grid: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        isotropic: bool = True
    ) -> np.ndarray:
        """Generate plateau kernel"""
        if grid is None:
            grid = self._mesh_grid(kernel_size)
        
        if isotropic:
            sigma_matrix = np.array([[sig_x**2, 0], [0, sig_x**2]])
        else:
            sigma_matrix = self._sigma_matrix2(sig_x, sig_y, theta)
        
        inverse_sigma = np.linalg.inv(sigma_matrix)
        kernel = np.reciprocal(
            np.power(
                inverse_sigma[0, 0] * (grid[0]**2) + 
                inverse_sigma[1, 1] * (grid[1]**2) + 
                2 * inverse_sigma[0, 1] * grid[0] * grid[1], 
                beta
            ) + 1
        )
        kernel = kernel / np.sum(kernel)
        return kernel
    
    def _random_mixed_kernels(
        self,
        kernel_list: List[str],
        kernel_prob: List[float],
        kernel_size: int,
        sigma_x_range: List[float],
        sigma_y_range: List[float],
        rotation_range: List[float],
        betag_range: List[float],
        betap_range: List[float],
        noise_range: Optional[List[float]] = None
    ) -> np.ndarray:
        """
        Randomly generate mixed blur kernels
        
        Args:
            kernel_list: List of kernel types
            kernel_prob: Probabilities for each kernel type
            kernel_size: Kernel size
            sigma_x_range: Sigma range in x direction
            sigma_y_range: Sigma range in y direction
            rotation_range: Rotation angle range
            betag_range: Generalized Gaussian beta range
            betap_range: Plateau kernel beta range
            noise_range: Noise range
        
        Returns:
            Generated blur kernel
        """
        kernel_type = random.choices(kernel_list, kernel_prob)[0]
        
        # Generate random parameters
        sigma_x = random.uniform(sigma_x_range[0], sigma_x_range[1])
        sigma_y = random.uniform(sigma_y_range[0], sigma_y_range[1])
        rotation = random.uniform(rotation_range[0], rotation_range[1])
        
        # Generate based on kernel type
        if kernel_type == 'iso':
            kernel = self._bivariate_gaussian(
                kernel_size, sigma_x, sigma_x, 0, isotropic=True
            )
        elif kernel_type == 'aniso':
            kernel = self._bivariate_gaussian(
                kernel_size, sigma_x, sigma_y, rotation, isotropic=False
            )
        elif kernel_type == 'generalized_iso':
            beta = random.uniform(betag_range[0], betag_range[1])
            kernel = self._bivariate_generalized_gaussian(
                kernel_size, sigma_x, sigma_x, 0, beta, isotropic=True
            )
        elif kernel_type == 'generalized_aniso':
            beta = random.uniform(betag_range[0], betag_range[1])
            kernel = self._bivariate_generalized_gaussian(
                kernel_size, sigma_x, sigma_y, rotation, beta, isotropic=False
            )
        elif kernel_type == 'plateau_iso':
            beta = random.uniform(betap_range[0], betap_range[1])
            kernel = self._bivariate_plateau(
                kernel_size, sigma_x, sigma_x, 0, beta, isotropic=True
            )
        elif kernel_type == 'plateau_aniso':
            beta = random.uniform(betap_range[0], betap_range[1])
            kernel = self._bivariate_plateau(
                kernel_size, sigma_x, sigma_y, rotation, beta, isotropic=False
            )
        else:
            raise ValueError(f'Unknown kernel type: {kernel_type}')
        
        # Add noise (if specified)
        if noise_range is not None:
            noise_level = random.uniform(noise_range[0], noise_range[1])
            noise = np.random.randn(*kernel.shape) * noise_level
            kernel = kernel + noise
            kernel = np.clip(kernel, 0, None)
        
        # Normalize
        kernel = kernel / np.sum(kernel)
        return kernel
    
    def _circular_lowpass_kernel(
        self, 
        cutoff: float, 
        kernel_size: int, 
        pad_to: int = 0
    ) -> np.ndarray:
        """
        Generate circular low-pass filter (sinc kernel)
        
        Args:
            cutoff: Cutoff frequency
            kernel_size: Kernel size
            pad_to: Pad to size
        
        Returns:
            Sinc filter kernel
        """
        assert kernel_size % 2 == 1, 'Kernel size must be an odd number.'
        
        kernel = np.fromfunction(
            lambda x, y: cutoff * np.sinc(
                cutoff * np.sqrt(
                    (x - (kernel_size - 1) / 2)**2 + 
                    (y - (kernel_size - 1) / 2)**2
                ) / np.pi
            ),
            [kernel_size, kernel_size]
        )
        kernel = kernel / np.sum(kernel)
        
        if pad_to > kernel_size:
            pad_size = (pad_to - kernel_size) // 2
            kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))
        
        return kernel
    
    def generate_kernels(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate three kernels needed for degradation: first blur kernel, second blur kernel, final sinc kernel
        
        Returns:
            (kernel1, kernel2, sinc_kernel)
        """
        # Generate first blur kernel
        kernel_size = random.choice(self.kernel_range1)
        if np.random.uniform() < self.opts['sinc_prob']:
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel1 = self._circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel1 = self._random_mixed_kernels(
                self.opts['kernel_list'],
                self.opts['kernel_prob'],
                kernel_size,
                self.opts['blur_sigma'],
                self.opts['blur_sigma'],
                [-math.pi, math.pi],
                self.opts['betag_range'],
                self.opts['betap_range'],
                noise_range=None
            )
        
        # Pad first kernel
        pad_size = (self.opts['blur_kernel_size'] - kernel_size) // 2
        kernel1 = np.pad(kernel1, ((pad_size, pad_size), (pad_size, pad_size)))
        
        # Generate second blur kernel
        kernel_size = random.choice(self.kernel_range2)
        if np.random.uniform() < self.opts['sinc_prob2']:
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel2 = self._circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel2 = self._random_mixed_kernels(
                self.opts['kernel_list2'],
                self.opts['kernel_prob2'],
                kernel_size,
                self.opts['blur_sigma2'],
                self.opts['blur_sigma2'],
                [-math.pi, math.pi],
                self.opts['betag_range2'],
                self.opts['betap_range2'],
                noise_range=None
            )
        
        # Pad second kernel
        pad_size = (self.opts['blur_kernel_size2'] - kernel_size) // 2
        kernel2 = np.pad(kernel2, ((pad_size, pad_size), (pad_size, pad_size)))
        
        # Generate final sinc kernel
        if np.random.uniform() < self.opts['final_sinc_prob']:
            kernel_size = random.choice(self.kernel_range2)
            omega_c = np.random.uniform(np.pi / 3, np.pi)
            sinc_kernel = self._circular_lowpass_kernel(
                omega_c, kernel_size, pad_to=self.opts['blur_kernel_size2']
            )
        else:
            sinc_kernel = self.pulse_tensor.numpy()
        
        return kernel1, kernel2, sinc_kernel
    
    def _filter2D(self, img: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        """
        Apply 2D convolution filtering to image using given kernel
        
        Args:
            img: Input image tensor [B, C, H, W]
            kernel: Convolution kernel [H, W]
        
        Returns:
            Filtered image
        """
        k = kernel.size(-1)
        b, c, h, w = img.size()
        
        # Expand kernel to match number of channels
        kernel = kernel.view(1, 1, k, k).repeat(c, 1, 1, 1)
        
        # Calculate padding
        pad = (k - 1) // 2
        
        # Apply convolution
        return F.conv2d(img, kernel, padding=pad, groups=c)
    
    def _random_add_gaussian_noise_pt(
        self,
        img: torch.Tensor,
        sigma_range: List[float],
        gray_prob: float = 0,
        clip: bool = True,
        rounds: bool = False
    ) -> torch.Tensor:
        """
        Add random Gaussian noise
        
        Args:
            img: Input image tensor
            sigma_range: Noise intensity range
            gray_prob: Probability of grayscale noise
            clip: Whether to clip to [0,1]
            rounds: Whether to round values
        
        Returns:
            Image with added noise
        """
        sigma = random.uniform(sigma_range[0], sigma_range[1]) / 255.0
        
        # Generate noise
        if random.random() < gray_prob:
            # Grayscale noise
            noise = torch.randn(img.size(0), 1, img.size(2), img.size(3), 
                              dtype=img.dtype, device=img.device) * sigma
            noise = noise.repeat(1, img.size(1), 1, 1)
        else:
            # Color noise
            noise = torch.randn_like(img) * sigma
        
        out = img + noise
        
        if clip and rounds:
            out = torch.clamp((out * 255.0).round(), 0, 255) / 255.
        elif clip:
            out = torch.clamp(out, 0, 1)
        elif rounds:
            out = (out * 255.0).round() / 255.
        
        return out
    
    def _random_add_poisson_noise_pt(
        self,
        img: torch.Tensor,
        scale_range: List[float],
        gray_prob: float = 0,
        clip: bool = True,
        rounds: bool = False
    ) -> torch.Tensor:
        """
        Add random Poisson noise
        
        Args:
            img: Input image tensor
            scale_range: Scale range
            gray_prob: Probability of grayscale noise
            clip: Whether to clip
            rounds: Whether to round values
        
        Returns:
            Image with added noise
        """
        scale = random.uniform(scale_range[0], scale_range[1])
        
        # Ensure input image values are non-negative (Poisson distribution requires rate >= 0)
        img = torch.clamp(img, min=0)
        
        if random.random() < gray_prob:
            # Grayscale noise
            img_gray = img.mean(dim=1, keepdim=True)
            img_gray = torch.poisson(img_gray * 255.0 * scale) / scale / 255.0
            noise = img_gray.repeat(1, img.size(1), 1, 1) - img
            out = img + noise
        else:
            # Color noise
            out = torch.poisson(img * 255.0 * scale) / scale / 255.0
        
        if clip and rounds:
            out = torch.clamp((out * 255.0).round(), 0, 255) / 255.
        elif clip:
            out = torch.clamp(out, 0, 1)
        elif rounds:
            out = (out * 255.0).round() / 255.
        
        return out
    
    def _jpeg_compress(self, img: torch.Tensor, quality: torch.Tensor) -> torch.Tensor:
        """
        Simulate JPEG compression
        
        Args:
            img: Input image tensor
            quality: JPEG quality parameter
        
        Returns:
            Compressed image
        """
        # Here we use simplified JPEG compression simulation
        # In practice, DiffJPEG library can be used
        if self.jpeger is None:
            try:
                from basicsr.utils import DiffJPEG
                self.jpeger = DiffJPEG(differentiable=False)
            except ImportError:
                # If DiffJPEG is not available, use simple quantization simulation
                return img
        
        return self.jpeger(img, quality=quality)
    
    def degrade(
        self, 
        img_gt: torch.Tensor,
        kernel1: Optional[torch.Tensor] = None,
        kernel2: Optional[torch.Tensor] = None,
        sinc_kernel: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Apply degradation pipeline to high-quality image
        
        Args:
            img_gt: High-quality image tensor [B, C, H, W]
            kernel1: First blur kernel (optional, auto-generated if not provided)
            kernel2: Second blur kernel (optional)
            sinc_kernel: Final sinc kernel (optional)
        
        Returns:
            Dictionary containing 'lq' (low-quality image) and 'gt' (high-quality image)
        """
        # Generate kernels if not provided
        if kernel1 is None or kernel2 is None or sinc_kernel is None:
            k1, k2, sk = self.generate_kernels()
            kernel1 = torch.FloatTensor(k1) if kernel1 is None else kernel1
            kernel2 = torch.FloatTensor(k2) if kernel2 is None else kernel2
            sinc_kernel = torch.FloatTensor(sk) if sinc_kernel is None else sinc_kernel
        
        # Ensure on same device
        device = img_gt.device
        kernel1 = kernel1.to(device)
        kernel2 = kernel2.to(device)
        sinc_kernel = sinc_kernel.to(device)
        
        ori_h, ori_w = img_gt.size()[2:4]
        sf = self.degradation['sf']
        
        # ----------------------- First Degradation Process ----------------------- #
        # Blur
        out = self._filter2D(img_gt, kernel1)
        
        # Random resize
        updown_type = random.choices(
            ['up', 'down', 'keep'],
            self.degradation['resize_prob']
        )[0]
        
        if updown_type == 'up':
            scale = random.uniform(1, self.degradation['resize_range'][1])
        elif updown_type == 'down':
            scale = random.uniform(self.degradation['resize_range'][0], 1)
        else:
            scale = 1
        
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(out, scale_factor=scale, mode=mode)
        
        # Add noise
        gray_noise_prob = self.degradation['gray_noise_prob']
        if random.random() < self.degradation['gaussian_noise_prob']:
            out = self._random_add_gaussian_noise_pt(
                out,
                sigma_range=self.degradation['noise_range'],
                clip=True,
                rounds=False,
                gray_prob=gray_noise_prob
            )
        else:
            out = self._random_add_poisson_noise_pt(
                out,
                scale_range=self.degradation['poisson_scale_range'],
                gray_prob=gray_noise_prob,
                clip=True,
                rounds=False
            )
        
        # JPEG compression
        jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.degradation['jpeg_range'])
        out = torch.clamp(out, 0, 1)
        out = self._jpeg_compress(out, quality=jpeg_p)
        
        # ----------------------- Second Degradation Process ----------------------- #
        if random.random() < self.degradation['second_order_prob']:
            # Blur
            if random.random() < self.degradation['second_blur_prob']:
                out = self._filter2D(out, kernel2)
            
            # Random resize
            updown_type = random.choices(
                ['up', 'down', 'keep'],
                self.degradation['resize_prob2']
            )[0]
            
            if updown_type == 'up':
                scale = random.uniform(1, self.degradation['resize_range2'][1])
            elif updown_type == 'down':
                scale = random.uniform(self.degradation['resize_range2'][0], 1)
            else:
                scale = 1
            
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(
                out,
                size=(int(ori_h / sf * scale), int(ori_w / sf * scale)),
                mode=mode
            )
            
            # Add noise
            gray_noise_prob = self.degradation['gray_noise_prob2']
            if random.random() < self.degradation['gaussian_noise_prob2']:
                out = self._random_add_gaussian_noise_pt(
                    out,
                    sigma_range=self.degradation['noise_range2'],
                    clip=True,
                    rounds=False,
                    gray_prob=gray_noise_prob
                )
            else:
                out = self._random_add_poisson_noise_pt(
                    out,
                    scale_range=self.degradation['poisson_scale_range2'],
                    gray_prob=gray_noise_prob,
                    clip=True,
                    rounds=False
                )
        
        # ----------------------- JPEG Compression + Final Sinc Filter ----------------------- #
        # Randomly choose order
        if random.random() < 0.5:
            # Resize and sinc filter first, then JPEG compress
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(
                out,
                size=(ori_h // sf, ori_w // sf),
                mode=mode
            )
            out = self._filter2D(out, sinc_kernel)
            
            # JPEG compression
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.degradation['jpeg_range2'])
            out = torch.clamp(out, 0, 1)
            out = self._jpeg_compress(out, quality=jpeg_p)
        else:
            # JPEG compress first, then resize and sinc filter
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.degradation['jpeg_range2'])
            out = torch.clamp(out, 0, 1)
            out = self._jpeg_compress(out, quality=jpeg_p)
            
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(
                out,
                size=(ori_h // sf, ori_w // sf),
                mode=mode
            )
            out = self._filter2D(out, sinc_kernel)
        
        # Clip and round
        im_lq = torch.clamp((out * 255.0).round(), 0, 255) / 255.
        
        return {
            'lq': im_lq.contiguous(),
            'gt': img_gt
        }


def test_degradation():
    """Test degradation pipeline"""
    import os
    import glob
    import matplotlib.pyplot as plt
    
    # Get configuration file path
    config_path = Path(__file__).parent.parent / 'configs' / 'realesrgan_degradation.yaml'
    
    # Create degradation pipeline
    degrader = RealESRGANDegradation(str(config_path))
    
    # Read images from traindata directory
    traindata_dir = Path(__file__).parent.parent / 'results'
    image_files = glob.glob(str(traindata_dir / '*.png')) + glob.glob(str(traindata_dir / '*.jpg')) + glob.glob(str(traindata_dir / '*.jpeg'))
    
    if not image_files:
        print(f"Error: No image files found in {traindata_dir} directory")
        return None
    
    # Read first image for testing
    img_path = image_files[1]
    print(f"Reading image: {img_path}")
    
    # Use PIL to read image
    img_pil = Image.open(img_path).convert('RGB')
    
    # Convert to tensor [1, 3, H, W], value range [0, 1]
    to_tensor = transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor()])
    img_gt = to_tensor(img_pil).unsqueeze(0)
    
    print(f"Input image shape: {img_gt.shape}")
    print(f"Input image value range: [{img_gt.min():.3f}, {img_gt.max():.3f}]")
    
    # Apply degradation
    result = degrader.degrade(img_gt)
    
    print(f"Output LQ image shape: {result['lq'].shape}")
    print(f"Output LQ image value range: [{result['lq'].min():.3f}, {result['lq'].max():.3f}]")
    print(f"Degradation pipeline test successful!")
    
    # Display image comparison before and after degradation
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Convert to numpy array for display [C, H, W] -> [H, W, C]
    img_gt_np = img_gt.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img_lq_np = result['lq'].squeeze(0).permute(1, 2, 0).cpu().numpy()
    
    # Display original image
    axes[0].imshow(img_gt_np)
    axes[0].set_title('Original (GT)', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Display degraded image
    axes[1].imshow(img_lq_np)
    axes[1].set_title('Degraded (LQ)', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    plt.tight_layout()
    
    plt.show()

if __name__ == '__main__':
    test_degradation()
