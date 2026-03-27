#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Gradio Demo for LPNSR Image Super-resolution
"""

import sys
from pathlib import Path

import gradio as gr
import numpy as np
import torch

# Add LPNSR to path
sys.path.insert(0, str(Path(__file__).parent))

from inference import NoisePredictorInference

# Global variable to store the inference engine
inference_engine = None


def initialize_inference(device="cuda", num_steps=4, color_correction=True):
    """Initialize the inference engine"""
    global inference_engine

    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available in this environment, using CPU instead")
        device = "cpu"

    if inference_engine is None:
        config_path = Path(__file__).parent / "configs" / "inference.yaml"

        # Create inference engine
        inference_engine = NoisePredictorInference(str(config_path), device=device)

        # Update configuration based on user inputs
        inference_engine.num_steps = num_steps
        inference_engine.color_correction = color_correction

        print(f"Inference engine initialized with {num_steps} steps")

    return "✓ Model loaded successfully!"


def process_image(
    input_image, num_steps, color_correction, use_swinir, use_noise_predictor, seed
):
    """Process a single image for super-resolution"""
    global inference_engine

    # Initialize if needed
    if inference_engine is None:
        initialize_inference(num_steps=num_steps, color_correction=color_correction)

    # Update settings
    inference_engine.num_steps = num_steps
    inference_engine.color_correction = color_correction
    inference_engine.use_swinir = use_swinir
    inference_engine.use_noise_predictor = use_noise_predictor

    # Convert seed to int
    try:
        seed = int(seed)
    except ValueError:
        seed = 12345

    # Set seed
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Read image from filepath
    import cv2

    lr_image = cv2.imread(input_image)
    lr_image = cv2.cvtColor(lr_image, cv2.COLOR_BGR2RGB)
    lr_image = lr_image.astype(np.float32) / 255.0

    # Process the image
    sr_image = inference_engine.process_single_image(lr_image)

    # Convert back to uint8
    sr_image = (sr_image * 255.0).astype(np.uint8)

    # Save to temp file
    import tempfile
    import time

    temp_dir = Path(tempfile.gettempdir())
    output_path = temp_dir / f"lpnsr_output_{int(time.time())}.png"
    cv2.imwrite(str(output_path), cv2.cvtColor(sr_image, cv2.COLOR_RGB2BGR))

    return str(output_path), str(output_path)


# Title and descriptions
title = "LPNSR: Prior-Enhanced Diffusion Image Super-Resolution via LR-Guided Noise Prediction"

description = r"""
<b>Official Gradio Demo</b> for <b>LPNSR</b> (Prior-Enhanced Diffusion Image Super-Resolution via LR-Guided Noise Prediction).<br>
🔥 LPNSR achieves state-of-the-art image super-resolution results with advanced noise prediction and Pre-Upsampling methods.<br>
"""

article = r"""
📋 **Features**
- High-quality 4x image super-resolution
- Advanced noise prediction for better detail reconstruction
- SwinIR-based super-resolution for high-quality baseline
- Color correction for natural-looking results

💡 **Tips**
- Upload a low-resolution image to see the super-resolution result
- Adjust the number of steps (1-4) for quality vs. speed trade-off
- Enable SwinIR for better baseline quality
- Enable color correction for natural colors

⚡ **Performance**
- Recommended: 4 steps for best quality
- Fast: 1-2 steps for quick preview
- Default seed: 12345 (can be changed for different results)
"""

# Create Gradio interface
with gr.Blocks() as demo:
    gr.Markdown(f"# {title}")
    gr.Markdown(description)

    with gr.Tabs():
        # Single Image Tab
        with gr.Tab("Single Image"):
            with gr.Row():
                with gr.Column():
                    input_image = gr.Image(
                        type="filepath", label="Input: Low Resolution Image"
                    )
                    num_steps = gr.Dropdown(
                        choices=[1, 2, 3, 4], value=4, label="Number of Steps"
                    )
                    color_correction = gr.Checkbox(
                        value=True, label="Enable Color Correction"
                    )
                    use_swinir = gr.Checkbox(
                        value=True, label="Enable SwinIR Super-resolution"
                    )
                    use_noise_predictor = gr.Checkbox(
                        value=True, label="Enable Noise Predictor"
                    )
                    seed = gr.Number(value=12345, label="Random Seed")
                    process_btn = gr.Button("Process", variant="primary")

                with gr.Column():
                    output_image = gr.Image(
                        type="filepath", label="Output: High Resolution Image"
                    )
                    output_file = gr.File(label="Download Output")

            process_btn.click(
                fn=process_image,
                inputs=[
                    input_image,
                    num_steps,
                    color_correction,
                    use_swinir,
                    use_noise_predictor,
                    seed,
                ],
                outputs=[output_image, output_file],
            )

    gr.Markdown(article)


# Launch the demo
if __name__ == "__main__":
    print("Initializing LPNSR Gradio Demo...")

    print("\nLaunching Gradio interface...")
    demo.queue(max_size=5)
    demo.launch(share=False, server_name="127.0.0.1", server_port=7860)
