#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Super-Resolution Model Test Script
"""

import argparse
import csv
import json
import logging
import os
import subprocess
import sys
import warnings

import cv2
import numpy as np
import torch
import yaml

# Suppress third-party deprecation noise (clip / timm, pulled in by pyiqa)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*pkg_resources.*")
warnings.filterwarnings("ignore", message=".*timm.models.layers.*")

# Suppress the "triton not found" noise emitted by torch.utils.flop_counter
# and xformers (triggered by FID's InceptionV3 feature extraction).
# The xformers warning is printed via the root logger, so we also raise the
# root logger level to ERROR. This is safe for an evaluation script.
logging.getLogger("torch.utils.flop_counter").setLevel(logging.ERROR)
logging.getLogger("xformers").setLevel(logging.ERROR)
logging.getLogger().setLevel(logging.ERROR)

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from infer import NoisePredictorInference


# ---------------------------------------------------------------------------
# pyiqa and lpips are hard dependencies: auto-install if missing.
# ---------------------------------------------------------------------------
def _ensure_deps():
    """Import pyiqa and lpips, auto-installing via pip if not available."""
    for pkg in ("pyiqa", "lpips"):
        try:
            __import__(pkg)
        except ImportError:
            print(f"{pkg} not found, installing via pip ...")
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", pkg]
            )


_ensure_deps()
import pyiqa


class MetricsCalculator:
    """
    Metrics Calculator

    Metric definitions follow RFMSR's `utils/cal_metrics.py`:
    - Full-reference metrics: PSNR (Y), SSIM (Y), LPIPS (Alex), DISTS
    - No-reference metrics: NIQE, MUSIQ, MANIQA, CLIPIQA
    - Dataset-level metric: FID
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

        # Metric parameters
        self.test_y_channel = config.get("test_y_channel", True)
        self.lpips_net = config.get("lpips_net", "alex")

        # Initialize metric calculators
        self._init_metrics()

    def _init_metrics(self):
        """Initialize metric calculators (pyiqa + lpips)."""
        self.metrics_enabled = {}
        self.metric_calculators = {}

        # ---- Full-reference metrics (pyiqa) ----
        # PSNR / SSIM: Y-channel via pyiqa's test_y_channel + color_space.
        if self.config.get("calculate_psnr", True):
            try:
                self.metric_calculators["psnr"] = pyiqa.create_metric(
                    "psnr",
                    test_y_channel=self.test_y_channel,
                    color_space="ycbcr",
                    device=self.device,
                )
                self.metrics_enabled["psnr"] = True
                print("  ✓ PSNR initialized (pyiqa, Y-channel)")
            except Exception as e:
                print(f"  ⚠ PSNR initialization failed: {e}")

        if self.config.get("calculate_ssim", True):
            try:
                self.metric_calculators["ssim"] = pyiqa.create_metric(
                    "ssim",
                    test_y_channel=self.test_y_channel,
                    color_space="ycbcr",
                    device=self.device,
                )
                self.metrics_enabled["ssim"] = True
                print("  ✓ SSIM initialized (pyiqa, Y-channel)")
            except Exception as e:
                print(f"  ⚠ SSIM initialization failed: {e}")

        if self.config.get("calculate_dists", True):
            try:
                self.metric_calculators["dists"] = pyiqa.create_metric(
                    "dists", device=self.device
                )
                self.metrics_enabled["dists"] = True
                print("  ✓ DISTS initialized (pyiqa)")
            except Exception as e:
                print(f"  ⚠ DISTS initialization failed: {e}")

        # ---- LPIPS (standalone lpips library, AlexNet, [-1,1] input) ----
        if self.config.get("calculate_lpips", True):
            try:
                import lpips

                self.metric_calculators["lpips"] = lpips.LPIPS(
                    net=self.lpips_net
                ).to(self.device)
                self.metrics_enabled["lpips"] = True
                print(f"  ✓ LPIPS initialized (lpips, net={self.lpips_net})")
            except Exception as e:
                print(f"  ⚠ LPIPS initialization failed: {e}")

        # ---- No-reference metrics (pyiqa, on SR only) ----
        for key, name in [
            ("niqe", "niqe"),
            ("musiq", "musiq"),
            ("maniqa", "maniqa"),
            ("clipiqa", "clipiqa"),
        ]:
            if not self.config.get(f"calculate_{key}", True):
                continue
            try:
                self.metric_calculators[key] = pyiqa.create_metric(
                    name, device=self.device
                )
                self.metrics_enabled[key] = True
                print(f"  ✓ {key.upper()} initialized (pyiqa)")
            except Exception as e:
                print(f"  ⚠ {key.upper()} initialization failed: {e}")

        # ---- FID (dataset-level, initialized lazily) ----
        if self.config.get("calculate_fid", True):
            try:
                self.metric_calculators["fid"] = pyiqa.create_metric(
                    "fid", device=self.device
                )
                self.metrics_enabled["fid"] = True
                print("  ✓ FID initialized (pyiqa)")
            except Exception as e:
                print(f"  ⚠ FID initialization failed: {e}")

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

    def _call_metric(self, key: str, *args) -> float:
        """Call a metric and return a scalar float."""
        with torch.no_grad():
            val = self.metric_calculators[key](*args)
        if torch.is_tensor(val):
            val = val.mean().item() if val.numel() > 1 else val.item()
        return float(val)

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
        sr_tensor = self._to_tensor(sr_img)   # [1, 3, H, W], [0, 1], RGB
        gt_tensor = self._to_tensor(gt_img)

        # PSNR / SSIM: pyiqa handles Y-channel conversion internally.
        for key in ("psnr", "ssim"):
            if self.metrics_enabled.get(key, False):
                try:
                    results[key] = self._call_metric(key, sr_tensor, gt_tensor)
                except Exception as e:
                    print(f"  ⚠ {key.upper()} calculation failed: {e}")
                    results[key] = float("nan")

        # LPIPS (standalone lpips library) expects [-1, 1] input.
        if self.metrics_enabled.get("lpips", False):
            try:
                gt_norm = (gt_tensor - 0.5) / 0.5
                sr_norm = (sr_tensor - 0.5) / 0.5
                results["lpips"] = self._call_metric("lpips", gt_norm, sr_norm)
            except Exception as e:
                print(f"  ⚠ LPIPS calculation failed: {e}")
                results["lpips"] = float("nan")

        # DISTS (pyiqa, full-reference).
        if self.metrics_enabled.get("dists", False):
            try:
                results["dists"] = self._call_metric("dists", sr_tensor, gt_tensor)
            except Exception as e:
                print(f"  ⚠ DISTS calculation failed: {e}")
                results["dists"] = float("nan")

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

        for key in ("niqe", "musiq", "maniqa", "clipiqa"):
            if self.metrics_enabled.get(key, False):
                try:
                    results[key] = self._call_metric(key, sr_tensor)
                except Exception as e:
                    print(f"  ⚠ {key.upper()} calculation failed: {e}")
                    results[key] = float("nan")

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

    def calculate_fid(self, sr_dir: str, gt_dir: str) -> float:
        """Calculate dataset-level FID between SR and GT directories."""
        if not self.metrics_enabled.get("fid", False):
            return float("nan")
        try:
            with torch.no_grad():
                fid = self.metric_calculators["fid"](sr_dir, gt_dir)
            return float(fid)
        except Exception as e:
            print(f"  ⚠ FID calculation failed: {e}")
            return float("nan")


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

        # Get LQ images (flat, non-recursive)
        lq_paths = []
        for ext in extensions:
            lq_paths.extend(lq_folder.glob(ext))
        lq_paths = sorted(list(set(lq_paths)))

        if len(lq_paths) == 0:
            raise ValueError(f"No image files found in {lq_folder}")

        # Match GT images by exact filename (strict, flat).
        pairs = []
        if self.has_gt:
            gt_folder = Path(self.gt_folder)
            for lq_path in lq_paths:
                gt_path = gt_folder / lq_path.name
                if gt_path.exists():
                    pairs.append((lq_path, gt_path))
                else:
                    pairs.append((lq_path, None))
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
        pbar = tqdm(pairs, desc="Testing progress")
        for lq_path, gt_path in pbar:
            result = OrderedDict()
            result["image_name"] = lq_path.name

            # Generate SR image
            sr_img = self._process_single_image(lq_path)

            # Save SR image (same filename as LQ, so it pairs with GT for FID)
            if self.save_sr_images:
                sr_save_path = sr_output_folder / f"{lq_path.stem}.png"
                cv2.imwrite(str(sr_save_path), sr_img)

            # Load GT image
            gt_img = None
            if gt_path is not None and gt_path.exists():
                gt_img = cv2.imread(str(gt_path))
                # Ensure GT and SR have same size
                if gt_img.shape[:2] != sr_img.shape[:2]:
                    pbar.write(
                        f"  ⚠ Size mismatch: SR={sr_img.shape[:2]}, GT={gt_img.shape[:2]}, skipping full-reference metrics"
                    )
                    gt_img = None

            # Calculate metrics
            metrics = self.metrics_calculator.calculate_all(sr_img, gt_img)
            result.update(metrics)

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

        # FID is a dataset-level metric computed once over SR/GT directories.
        if self.has_gt and self.save_sr_images:
            fid_val = self.metrics_calculator.calculate_fid(
                str(sr_output_folder), str(self.gt_folder)
            )
            if not np.isnan(fid_val):
                avg_results["fid"] = fid_val
                print(f"  {'FID':10s}: {fid_val:.4f}")

        # Save results to CSV
        if self.save_metrics_csv:
            csv_path = self.output_folder / "metrics.csv"
            # fieldnames = per-image fields + dataset-level fields (e.g. fid)
            fieldnames = list(all_results[0].keys())
            for k in avg_results:
                if k not in fieldnames:
                    fieldnames.append(k)
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
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
