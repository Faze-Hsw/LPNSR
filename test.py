#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Super-Resolution Model Test Script
"""

import argparse
import csv
import json
import os
import sys

import cv2
import numpy as np
import torch
import yaml

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from inference import NoisePredictorInference


class MetricsCalculator:
    """
    Metrics Calculator

    Supported metrics:
    - Full-reference metrics: PSNR, SSIM, LPIPS
    - No-reference metrics: NIQE, PI, CLIP-IQA, MUSIQ
    """

    def __init__(self, config: dict, device: str = "cuda"):
        """
        Initialize metrics calculator

        Args:
            config: Metrics configuration
            device: Computing device
        """
        self.config = config
        self.device = device
        self.use_pyiqa = config.get("use_pyiqa", True)

        # Metric parameters
        self.crop_border = config.get("crop_border", 4)
        self.test_y_channel = config.get("test_y_channel", True)
        self.lpips_net = config.get("lpips_net", "alex")

        # Initialize metric calculators
        self._init_metrics()

    def _init_metrics(self):
        """Initialize various metric calculators"""
        self.metrics_enabled = {}
        self.metric_calculators = {}

        # Check if pyiqa library is available
        self.pyiqa_available = False
        if self.use_pyiqa:
            try:
                import pyiqa

                self.pyiqa_available = True
                print("✓ pyiqa library available, using official implementation")
            except ImportError:
                print("⚠ pyiqa library not available, using built-in implementation")
                print("  Recommended: pip install pyiqa")

        # Initialize full-reference metrics
        # PSNR
        if self.config.get("calculate_psnr", True):
            self.metrics_enabled["psnr"] = True
            from metrics.psnr import PSNR

            self.metric_calculators["psnr"] = PSNR(
                crop_border=self.crop_border, test_y_channel=self.test_y_channel
            )
            print("  ✓ PSNR initialized")

        # SSIM
        if self.config.get("calculate_ssim", True):
            self.metrics_enabled["ssim"] = True
            from metrics.ssim import SSIM

            self.metric_calculators["ssim"] = SSIM(
                crop_border=self.crop_border, test_y_channel=self.test_y_channel
            )
            print("  ✓ SSIM initialized")

        # LPIPS
        if self.config.get("calculate_lpips", True):
            self.metrics_enabled["lpips"] = True
            if self.pyiqa_available:
                import pyiqa

                self.metric_calculators["lpips"] = pyiqa.create_metric(
                    "lpips", device=self.device, net=self.lpips_net
                )
            else:
                from metrics.lpips import LPIPS

                self.metric_calculators["lpips"] = LPIPS(
                    net=self.lpips_net, use_gpu=(self.device == "cuda")
                )
            print(f"  ✓ LPIPS initialized (net={self.lpips_net})")

        # Initialize no-reference metrics
        # NIQE
        if self.config.get("calculate_niqe", True):
            self.metrics_enabled["niqe"] = True
            if self.pyiqa_available:
                import pyiqa

                self.metric_calculators["niqe"] = pyiqa.create_metric(
                    "niqe", device=self.device
                )
            else:
                from metrics.niqe import NIQE

                self.metric_calculators["niqe"] = NIQE(device=self.device)
            print("  ✓ NIQE initialized")

        # PI
        if self.config.get("calculate_pi", True):
            self.metrics_enabled["pi"] = True
            if self.pyiqa_available:
                try:
                    import pyiqa

                    self.metric_calculators["pi"] = pyiqa.create_metric(
                        "pi", device=self.device
                    )
                except Exception as e:
                    print(
                        f"  ⚠ PI (pyiqa) initialization failed: {e}, using built-in implementation"
                    )
                    from metrics.pi import PI

                    self.metric_calculators["pi"] = PI(device=self.device)
            else:
                from metrics.pi import PI

                self.metric_calculators["pi"] = PI(device=self.device)
            print("  ✓ PI initialized")

        # CLIP-IQA
        if self.config.get("calculate_clipiqa", True):
            self.metrics_enabled["clipiqa"] = True
            if self.pyiqa_available:
                try:
                    import pyiqa

                    self.metric_calculators["clipiqa"] = pyiqa.create_metric(
                        "clipiqa", device=self.device
                    )
                except Exception as e:
                    print(f"  ⚠ CLIP-IQA (pyiqa) initialization failed: {e}")
                    self.metrics_enabled["clipiqa"] = False
            else:
                from metrics.clipiqa import CLIPIQA

                self.metric_calculators["clipiqa"] = CLIPIQA(device=self.device)
            if self.metrics_enabled["clipiqa"]:
                print("  ✓ CLIP-IQA initialized")

        # MUSIQ
        if self.config.get("calculate_musiq", True):
            self.metrics_enabled["musiq"] = True
            if self.pyiqa_available:
                try:
                    import pyiqa

                    self.metric_calculators["musiq"] = pyiqa.create_metric(
                        "musiq", device=self.device
                    )
                except Exception as e:
                    print(f"  ⚠ MUSIQ (pyiqa) initialization failed: {e}")
                    self.metrics_enabled["musiq"] = False
            else:
                from metrics.musiq import MUSIQ

                self.metric_calculators["musiq"] = MUSIQ(device=self.device)
            if self.metrics_enabled["musiq"]:
                print("  ✓ MUSIQ initialized")

    def _to_tensor(self, img: np.ndarray) -> torch.Tensor:
        """
        Convert numpy image to PyTorch tensor

        Args:
            img: numpy image (H, W, C), [0, 255], BGR

        Returns:
            Tensor (1, C, H, W), [0, 1], RGB
        """
        # BGR -> RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # HWC -> CHW
        img_chw = img_rgb.transpose(2, 0, 1)
        # Normalize to [0, 1]
        img_tensor = torch.from_numpy(img_chw).float() / 255.0
        # Add batch dimension
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        return img_tensor

    def calculate_fr_metrics(self, sr_img: np.ndarray, gt_img: np.ndarray) -> dict:
        """
        Calculate full-reference metrics

        Args:
            sr_img: SR image (H, W, C), [0, 255], BGR
            gt_img: GT image (H, W, C), [0, 255], BGR

        Returns:
            Metrics dictionary
        """
        results = {}

        # PSNR
        if self.metrics_enabled.get("psnr", False):
            psnr_val = self.metric_calculators["psnr"](sr_img, gt_img)
            results["psnr"] = psnr_val

        # SSIM
        if self.metrics_enabled.get("ssim", False):
            ssim_val = self.metric_calculators["ssim"](sr_img, gt_img)
            results["ssim"] = ssim_val

        # LPIPS
        if self.metrics_enabled.get("lpips", False):
            sr_tensor = self._to_tensor(sr_img)
            gt_tensor = self._to_tensor(gt_img)

            with torch.no_grad():
                if self.pyiqa_available:
                    lpips_val = self.metric_calculators["lpips"](
                        sr_tensor, gt_tensor
                    ).item()
                else:
                    lpips_val = self.metric_calculators["lpips"](sr_tensor, gt_tensor)
            results["lpips"] = lpips_val

        return results

    def calculate_nr_metrics(self, sr_img: np.ndarray) -> dict:
        """
        Calculate no-reference metrics

        Args:
            sr_img: SR image (H, W, C), [0, 255], BGR

        Returns:
            Metrics dictionary
        """
        results = {}
        sr_tensor = self._to_tensor(sr_img)

        # NIQE
        if self.metrics_enabled.get("niqe", False):
            with torch.no_grad():
                if self.pyiqa_available and hasattr(
                    self.metric_calculators["niqe"], "__call__"
                ):
                    niqe_val = self.metric_calculators["niqe"](sr_tensor).item()
                else:
                    niqe_val = self.metric_calculators["niqe"](sr_img)
            results["niqe"] = niqe_val

        # PI
        if self.metrics_enabled.get("pi", False):
            with torch.no_grad():
                try:
                    if self.pyiqa_available and hasattr(
                        self.metric_calculators["pi"], "__call__"
                    ):
                        pi_val = self.metric_calculators["pi"](sr_tensor).item()
                    else:
                        pi_val = self.metric_calculators["pi"](sr_img)
                    results["pi"] = pi_val
                except Exception as e:
                    print(f"  ⚠ PI calculation failed: {e}")
                    results["pi"] = float("nan")

        # CLIP-IQA
        if self.metrics_enabled.get("clipiqa", False):
            with torch.no_grad():
                try:
                    if self.pyiqa_available:
                        clipiqa_val = self.metric_calculators["clipiqa"](
                            sr_tensor
                        ).item()
                    else:
                        clipiqa_val = self.metric_calculators["clipiqa"](sr_img)
                    results["clipiqa"] = clipiqa_val
                except Exception as e:
                    print(f"  ⚠ CLIP-IQA calculation failed: {e}")
                    results["clipiqa"] = float("nan")

        # MUSIQ
        if self.metrics_enabled.get("musiq", False):
            with torch.no_grad():
                try:
                    if self.pyiqa_available:
                        musiq_val = self.metric_calculators["musiq"](sr_tensor).item()
                    else:
                        musiq_val = self.metric_calculators["musiq"](sr_img)
                    results["musiq"] = musiq_val
                except Exception as e:
                    print(f"  ⚠ MUSIQ calculation failed: {e}")
                    results["musiq"] = float("nan")

        return results

    def calculate_all(self, sr_img: np.ndarray, gt_img: np.ndarray = None) -> dict:
        """
        Calculate all metrics

        Args:
            sr_img: SR image
            gt_img: GT image (optional, if not provided, only no-reference metrics are calculated)

        Returns:
            Dictionary of all metrics
        """
        results = {}

        # Calculate full-reference metrics
        if gt_img is not None:
            fr_results = self.calculate_fr_metrics(sr_img, gt_img)
            results.update(fr_results)

        # Calculate no-reference metrics
        nr_results = self.calculate_nr_metrics(sr_img)
        results.update(nr_results)

        return results


class SRTester:
    """
    Super-Resolution Tester

    Features:
    1. Load LQ images and generate SR images
    2. Calculate various evaluation metrics
    3. Save results
    """

    def __init__(self, config_path: str):
        """
        Initialize tester

        Args:
            config_path: Test config file path
        """
        # Load config
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        # Set random seed
        seed = self.config.get("seed", 12345)
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Device
        self.device = self.config.get("device", "cuda")
        if self.device == "cuda" and not torch.cuda.is_available():
            print("⚠ CUDA not available, using CPU")
            self.device = "cpu"

        # Data paths
        self.gt_folder = self.config["data"].get("gt_folder", "")
        self.lq_folder = self.config["data"]["lq_folder"]
        self.output_folder = Path(
            self.config["data"].get("output_folder", "./test_results")
        )

        # Check if GT images exist
        self.has_gt = bool(self.gt_folder) and Path(self.gt_folder).exists()

        # Output config
        self.output_config = self.config.get("output", {})
        self.save_sr_images = self.output_config.get("save_sr_images", True)
        self.save_metrics_csv = self.output_config.get("save_metrics_csv", True)
        self.save_metrics_json = self.output_config.get("save_metrics_json", True)
        self.print_per_image = self.output_config.get("print_per_image", True)

        # Initialize inferencer
        print("\n" + "=" * 60)
        print("Initializing inferencer...")
        print("=" * 60)
        inference_config = self.config["inference"]["config_path"]
        self.inferencer = NoisePredictorInference(inference_config, device=self.device)

        # Initialize metrics calculator
        print("\n" + "=" * 60)
        print("Initializing metrics calculator...")
        print("=" * 60)
        self.metrics_calculator = MetricsCalculator(
            self.config["metrics"], device=self.device
        )

        print("\n" + "=" * 60)
        print("Tester initialized")
        print("=" * 60)
        print(f"  - LQ folder: {self.lq_folder}")
        print(
            f"  - GT folder: {self.gt_folder if self.has_gt else 'None (only no-reference metrics)'}"
        )
        print(f"  - Output folder: {self.output_folder}")
        print(f"  - Device: {self.device}")

    def _get_image_pairs(self) -> list:
        """
        Get list of image pairs

        Returns:
            [(lq_path, gt_path), ...] list, gt_path may be None
        """
        lq_folder = Path(self.lq_folder)

        # Supported image formats
        extensions = [
            "*.png",
            "*.jpg",
            "*.jpeg",
            "*.bmp",
            "*.PNG",
            "*.JPG",
            "*.JPEG",
            "*.BMP",
        ]

        # Get LQ images
        lq_paths = []
        for ext in extensions:
            lq_paths.extend(lq_folder.glob(ext))
        lq_paths = sorted(list(set(lq_paths)))

        if len(lq_paths) == 0:
            raise ValueError(f"No image files found in {lq_folder}")

        # Match GT images
        pairs = []
        if self.has_gt:
            gt_folder = Path(self.gt_folder)
            for lq_path in lq_paths:
                # Try different matching methods
                gt_path = None

                # 1. Exact same filename
                candidate = gt_folder / lq_path.name
                if candidate.exists():
                    gt_path = candidate
                else:
                    # 2. Remove suffix like _lq, _lr, x4, etc.
                    stem = lq_path.stem
                    for suffix in [
                        "_lq",
                        "_lr",
                        "_LQ",
                        "_LR",
                        "x4",
                        "x2",
                        "_bicubic",
                        "_LR4",
                    ]:
                        if stem.endswith(suffix):
                            stem = stem[: -len(suffix)]
                            break

                    # Try different extensions
                    for ext in [".png", ".jpg", ".jpeg", ".bmp"]:
                        candidate = gt_folder / (stem + ext)
                        if candidate.exists():
                            gt_path = candidate
                            break
                        # Try adding _gt, _hr suffix
                        for gt_suffix in ["_gt", "_hr", "_GT", "_HR"]:
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
        Process a single image

        Args:
            lq_path: LQ image path

        Returns:
            SR image (H, W, C), [0, 255], BGR
        """
        # Read LQ image
        lq_img = cv2.imread(str(lq_path))
        lq_img_rgb = cv2.cvtColor(lq_img, cv2.COLOR_BGR2RGB)
        lq_img_float = lq_img_rgb.astype(np.float32) / 255.0

        # Super-resolution processing
        sr_img_float = self.inferencer.process_single_image(lq_img_float)

        # Convert to uint8 BGR
        sr_img = (sr_img_float * 255.0).clip(0, 255).astype(np.uint8)
        sr_img_bgr = cv2.cvtColor(sr_img, cv2.COLOR_RGB2BGR)

        return sr_img_bgr

    def run(self):
        """
        Run testing
        """
        print("\n" + "=" * 60)
        print("Starting test")
        print("=" * 60)

        # Create output directory
        self.output_folder.mkdir(parents=True, exist_ok=True)
        sr_output_folder = self.output_folder / "sr_images"
        if self.save_sr_images:
            sr_output_folder.mkdir(parents=True, exist_ok=True)

        # Get image pairs
        pairs = self._get_image_pairs()
        print(f"\nFound {len(pairs)} images")

        if self.has_gt:
            gt_count = sum(1 for _, gt in pairs if gt is not None)
            print(f"  - Matched GT images: {gt_count}")

        # Store all results
        all_results = []

        # Process each image
        for lq_path, gt_path in tqdm(pairs, desc="Testing progress"):
            result = OrderedDict()
            result["image_name"] = lq_path.name

            # Generate SR image
            sr_img = self._process_single_image(lq_path)

            # Save SR image
            if self.save_sr_images:
                sr_save_path = sr_output_folder / f"{lq_path.stem}_sr.png"
                cv2.imwrite(str(sr_save_path), sr_img)

            # Load GT image
            gt_img = None
            if gt_path is not None and gt_path.exists():
                gt_img = cv2.imread(str(gt_path))
                # Ensure GT and SR have same size
                if gt_img.shape[:2] != sr_img.shape[:2]:
                    print(
                        f"  ⚠ Size mismatch: SR={sr_img.shape[:2]}, GT={gt_img.shape[:2]}, skipping full-reference metrics"
                    )
                    gt_img = None

            # Calculate metrics
            metrics = self.metrics_calculator.calculate_all(sr_img, gt_img)
            result.update(metrics)

            # Print per-image result
            if self.print_per_image:
                metrics_str = ", ".join(
                    [f"{k}: {v:.4f}" for k, v in metrics.items() if not np.isnan(v)]
                )
                print(f"\n  {lq_path.name}: {metrics_str}")

            all_results.append(result)

        # Calculate average
        print("\n" + "=" * 60)
        print("Test Results Summary")
        print("=" * 60)

        avg_results = OrderedDict()
        metric_keys = [k for k in all_results[0].keys() if k != "image_name"]

        for key in metric_keys:
            values = [
                r[key]
                for r in all_results
                if key in r and not np.isnan(r.get(key, float("nan")))
            ]
            if values:
                avg_results[key] = np.mean(values)
                std = np.std(values)
                print(f"  {key.upper():10s}: {avg_results[key]:.4f} ± {std:.4f}")

        # Save results to CSV
        if self.save_metrics_csv:
            csv_path = self.output_folder / "metrics.csv"
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
                writer.writeheader()
                writer.writerows(all_results)
                # Add average row
                avg_row = {"image_name": "AVERAGE"}
                avg_row.update(avg_results)
                writer.writerow(avg_row)
            print(f"\n✓ Metrics saved to: {csv_path}")

        # Save results to JSON
        if self.save_metrics_json:
            json_path = self.output_folder / "metrics.json"
            output_data = {
                "timestamp": datetime.now().isoformat(),
                "config": {
                    "lq_folder": str(self.lq_folder),
                    "gt_folder": str(self.gt_folder) if self.has_gt else None,
                    "metrics_config": self.config["metrics"],
                },
                "average": {k: float(v) for k, v in avg_results.items()},
                "per_image": [
                    {
                        k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                        for k, v in r.items()
                    }
                    for r in all_results
                ],
            }
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            print(f"✓ Metrics saved to: {json_path}")

        if self.save_sr_images:
            print(f"✓ SR images saved to: {sr_output_folder}")

        print("\n" + "=" * 60)
        print("Test completed!")
        print("=" * 60)

        return avg_results


def get_parser():
    """Get command line argument parser"""
    parser = argparse.ArgumentParser(description="Super-Resolution Model Test Script")

    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="configs/test_config.yaml",
        help="Test config file path",
    )
    parser.add_argument(
        "--lq_folder",
        type=str,
        default=None,
        help="LQ image folder path (overrides config file)",
    )
    parser.add_argument(
        "--gt_folder",
        type=str,
        default=None,
        help="GT image folder path (overrides config file)",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default=None,
        help="Output folder path (overrides config file)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu"],
        help="Computing device (overrides config file)",
    )
    parser.add_argument(
        "--no_save_sr", action="store_true", help="Do not save SR images"
    )

    return parser


def main():
    """Main function"""
    parser = get_parser()
    args = parser.parse_args()

    print("=" * 60)
    print("Super-Resolution Model Test Script")
    print("=" * 60)

    # Load config
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Command line arguments override config
    if args.lq_folder:
        config["data"]["lq_folder"] = args.lq_folder
    if args.gt_folder:
        config["data"]["gt_folder"] = args.gt_folder
    if args.output_folder:
        config["data"]["output_folder"] = args.output_folder
    if args.device:
        config["device"] = args.device
    if args.no_save_sr:
        config["output"]["save_sr_images"] = False

    # Check required parameters
    if not config["data"].get("lq_folder"):
        raise ValueError(
            "LQ image folder path must be specified (via config file or --lq_folder argument)"
        )

    # Temporarily save modified config
    temp_config_path = Path(args.config).parent / "test_config_temp.yaml"
    with open(temp_config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True)

    try:
        # Create tester and run
        tester = SRTester(str(temp_config_path))
        results = tester.run()
    finally:
        # Clean up temporary file
        if temp_config_path.exists():
            temp_config_path.unlink()

    return results


if __name__ == "__main__":
    main()
