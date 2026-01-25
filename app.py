#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Gradio Demo for LPNSR Image Super-resolution
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import gradio as gr
from pathlib import Path
import sys

# Add LPNSR to path
sys.path.insert(0, str(Path(__file__).parent))

from inference import NoisePredictorInference


# Global variable to store the inference engine
inference_engine = None


def initialize_inference(device='cuda', num_steps=4, color_correction=True):
    """Initialize the inference engine"""
    global inference_engine

    if inference_engine is None:
        config_path = Path(__file__).parent / 'configs' / 'inference.yaml'

        # Create inference engine
        inference_engine = NoisePredictorInference(str(config_path), device=device)

        # Update configuration based on user inputs
        inference_engine.num_steps = num_steps
        inference_engine.color_correction = color_correction

        print(f"Inference engine initialized with {num_steps} steps")

    return "✓ Model loaded successfully!"


def process_image(input_image, num_steps, color_correction, use_initializer, use_noise_predictor, seed):
    """Process a single image for super-resolution"""
    global inference_engine

    # Initialize if needed
    if inference_engine is None:
        initialize_inference(num_steps=num_steps, color_correction=color_correction)

    # Update settings
    inference_engine.num_steps = num_steps
    inference_engine.color_correction = color_correction
    inference_engine.use_initializer = use_initializer
    inference_engine.use_noise_predictor = use_noise_predictor

    # Convert seed to int
    try:
        seed = int(seed)
    except:
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


def process_batch(input_dir, output_dir, num_steps, color_correction, use_initializer, use_noise_predictor, seed):
    """Process a batch of images"""
    global inference_engine

    # Initialize if needed
    if inference_engine is None:
        initialize_inference(num_steps=num_steps, color_correction=color_correction)

    # Update settings
    inference_engine.num_steps = num_steps
    inference_engine.color_correction = color_correction
    inference_engine.use_initializer = use_initializer
    inference_engine.use_noise_predictor = use_noise_predictor

    # Convert seed to int
    try:
        seed = int(seed)
    except:
        seed = 12345

    # Set seed
    import torch
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Get input path
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    # Get image files
    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.PNG', '*.JPG', '*.JPEG']:
        image_files.extend(input_path.glob(ext))

    total_files = len(image_files)

    if total_files == 0:
        return f"No image files found in {input_dir}"

    # Process each image
    for idx, img_path in enumerate(image_files):
        # Read image
        import cv2
        lr_image = cv2.imread(str(img_path))
        lr_image = cv2.cvtColor(lr_image, cv2.COLOR_BGR2RGB)
        lr_image = lr_image.astype(np.float32) / 255.0

        # Process
        sr_image = inference_engine.process_single_image(lr_image)

        # Save result
        sr_image = (sr_image * 255.0).astype(np.uint8)
        sr_image = cv2.cvtColor(sr_image, cv2.COLOR_RGB2BGR)

        output_file = output_path / f"{img_path.stem}_sr.png"
        cv2.imwrite(str(output_file), sr_image)

    return f"✓ Processed {total_files} images. Results saved in {output_path}"


# Title and descriptions
title = "LPNSR: Image Super-resolution via Latent Proximal Noise Sampling"

description = r"""
<b>Official Gradio Demo</b> for <b>LPNSR</b> (Latent Proximal Noise Sampling for Image Super-resolution).<br>
🔥 LPNSR achieves state-of-the-art image super-resolution results with advanced noise prediction and initialization techniques.<br>
"""

article = r"""
📋 **Features**
- High-quality 4x image super-resolution
- Advanced noise prediction for better detail reconstruction
- Intelligent initializer for faster convergence
- Color correction for natural-looking results
- Batch processing support

💡 **Tips**
- Upload a low-resolution image to see the super-resolution result
- Adjust the number of steps (1-4) for quality vs. speed trade-off
- Enable color correction for natural colors
- Use batch processing to process multiple images at once

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
                        type="filepath",
                        label="Input: Low Resolution Image"
                    )
                    num_steps = gr.Dropdown(
                        choices=[1, 2, 3, 4],
                        value=4,
                        label="Number of Steps"
                    )
                    color_correction = gr.Checkbox(
                        value=True,
                        label="Enable Color Correction"
                    )
                    use_initializer = gr.Checkbox(
                        value=True,
                        label="Enable Initializer"
                    )
                    use_noise_predictor = gr.Checkbox(
                        value=True,
                        label="Enable Noise Predictor"
                    )
                    seed = gr.Number(
                        value=12345,
                        label="Random Seed"
                    )
                    process_btn = gr.Button("Process", variant="primary")

                with gr.Column():
                    output_image = gr.Image(
                        type="filepath",
                        label="Output: High Resolution Image"
                    )
                    output_file = gr.File(
                        label="Download Output"
                    )

            process_btn.click(
                fn=process_image,
                inputs=[input_image, num_steps, color_correction, use_initializer, use_noise_predictor, seed],
                outputs=[output_image, output_file]
            )

        # Batch Processing Tab
        with gr.Tab("Batch Processing"):
            with gr.Column():
                input_dir = gr.Textbox(
                    label="Input Directory Path",
                    placeholder="e.g., /path/to/input/images"
                )
                output_dir = gr.Textbox(
                    label="Output Directory Path",
                    placeholder="e.g., /path/to/output/images",
                    value="./batch_results"
                )
                batch_num_steps = gr.Dropdown(
                    choices=[1, 2, 3, 4],
                    value=4,
                    label="Number of Steps"
                )
                batch_color_correction = gr.Checkbox(
                    value=True,
                    label="Enable Color Correction"
                )
                batch_use_initializer = gr.Checkbox(
                    value=True,
                    label="Enable Initializer"
                )
                batch_use_noise_predictor = gr.Checkbox(
                    value=True,
                    label="Enable Noise Predictor"
                )
                batch_seed = gr.Number(
                    value=12345,
                    label="Random Seed"
                )
                batch_btn = gr.Button("Process Folder", variant="primary")
                batch_status = gr.Textbox(
                    label="Processing Status",
                    interactive=False
                )

            batch_btn.click(
                fn=process_batch,
                inputs=[input_dir, output_dir, batch_num_steps, batch_color_correction, batch_use_initializer, batch_use_noise_predictor, batch_seed],
                outputs=batch_status
            )

    gr.Markdown(article)


# Launch the demo
if __name__ == "__main__":
    print("Initializing LPNSR Gradio Demo...")

    print("\nLaunching Gradio interface...")
    demo.queue(max_size=5)
    demo.launch(
        share=False,
        server_name="127.0.0.1",
        server_port=7860
    )
