#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TP1 Common I/O Functions

Shared utilities for loading/saving images, metadata, and color conversions.
"""

import numpy as np
import os
import json
from PIL import Image

# Try to use tifffile for better 16-bit support
try:
    import tifffile
    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False
    print("Warning: tifffile not installed. Using PIL for TIFF (may have precision issues).")


# =============================================================================
# TIFF I/O
# =============================================================================

def load_tiff(filepath):
    """
    Load a TIFF image and convert to normalized float32 [0, 1].
    
    Works for both grayscale (mosaic) and RGB images.
    """
    if HAS_TIFFFILE:
        image = tifffile.imread(filepath)
    else:
        image = np.array(Image.open(filepath))
    
    if image.dtype == np.uint16:
        return image.astype(np.float32) / 65535.0
    elif image.dtype == np.uint8:
        return image.astype(np.float32) / 255.0
    return image.astype(np.float32)


def save_tiff16(image, filepath):
    """
    Save image as 16-bit TIFF.
    
    Args:
        image: Normalized image [0,1] - 2D for mosaic, 3D for RGB
        filepath: Output path
    """
    img_clipped = np.clip(image, 0, 1)
    img_16bit = (img_clipped * 65535).astype(np.uint16)
    
    if HAS_TIFFFILE:
        if img_16bit.ndim == 3:
            tifffile.imwrite(filepath, img_16bit, photometric='rgb')
        else:
            tifffile.imwrite(filepath, img_16bit)
    else:
        if img_16bit.ndim == 3:
            Image.fromarray(img_16bit, mode='RGB').save(filepath)
        else:
            Image.fromarray(img_16bit, mode='I;16').save(filepath)
    
    print(f"  Saved TIFF: {filepath}")


# =============================================================================
# JSON Metadata I/O
# =============================================================================

def load_metadata(json_path):
    """Load metadata from JSON file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_metadata(metadata, filepath):
    """
    Save metadata to a JSON file.
    
    Handles numpy type conversion automatically.
    """
    def to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [to_serializable(i) for i in obj]
        return obj
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(to_serializable(metadata), f, indent=2, ensure_ascii=False)
    
    print(f"  Saved JSON: {filepath}")


# =============================================================================
# Color Space Conversions
# =============================================================================

def linear_to_srgb(linear):
    """
    Apply sRGB OETF (gamma encoding) for display.
    
    Formula:
    - If linear ≤ 0.0031308: sRGB = 12.92 × linear
    - Otherwise: sRGB = 1.055 × linear^(1/2.4) − 0.055
    """
    linear = np.clip(linear, 0, None)  # Allow > 1 before gamma
    srgb = np.where(
        linear <= 0.0031308,
        12.92 * linear,
        1.055 * np.power(linear, 1/2.4) - 0.055
    )
    return np.clip(srgb, 0, 1)


def xyz_to_linear_srgb(xyz_image):
    """
    Convert XYZ to linear sRGB colorspace (D65 illuminant).
    """
    xyz_to_rgb = np.array([
        [ 3.2406255, -1.5372080, -0.4986286],
        [-0.9689307,  1.8757561,  0.0415175],
        [ 0.0557101, -0.2040211,  1.0569959]
    ])
    
    H, W, _ = xyz_image.shape
    pixels = xyz_image.reshape(-1, 3)
    rgb_linear = pixels @ xyz_to_rgb.T
    return rgb_linear.reshape(H, W, 3)


def xyz_to_srgb(xyz_image):
    """Convert XYZ to sRGB (linear conversion + gamma)."""
    rgb_linear = xyz_to_linear_srgb(xyz_image)
    rgb_linear = np.clip(rgb_linear, 0, 1)
    return linear_to_srgb(rgb_linear)


# =============================================================================
# JPEG/PNG I/O
# =============================================================================

def save_jpeg(image, filepath, quality=95):
    """
    Save image as sRGB JPEG.
    
    Args:
        image: Linear RGB image [0,1]
        filepath: Output path
        quality: JPEG quality (1-100)
    
    Returns:
        8-bit sRGB image array
    """
    srgb = linear_to_srgb(image)
    srgb_8bit = (srgb * 255).astype(np.uint8)
    Image.fromarray(srgb_8bit, mode='RGB').save(filepath, 'JPEG', quality=quality)
    print(f"  Saved JPEG: {filepath}")
    return srgb_8bit


def save_png(image, filepath):
    """
    Save image as 8-bit PNG.
    
    Args:
        image: Either uint8 array or float [0,1] (will apply sRGB gamma)
        filepath: Output path
    """
    if image.dtype == np.uint8:
        img_8bit = image
    else:
        srgb = linear_to_srgb(np.clip(image, 0, 1))
        img_8bit = (srgb * 255).astype(np.uint8)
    
    Image.fromarray(img_8bit, mode='RGB').save(filepath, 'PNG')
    print(f"  Saved PNG: {filepath}")


def quantize_to_8bit(image_float):
    """Quantize float32 [0,1] image to 8-bit."""
    return np.clip(255 * image_float, 0, 255).astype(np.uint8)

