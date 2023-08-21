import os
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from config import ex, config
from Dataset import AnimalKingdomDatasetVisualize

def draw_patches(image_array, gap=10, gap_transparent=False):
    # Check if the image dimensions are divisible by 4
    if image_array.shape[0] % 4 != 0 or image_array.shape[1] % 4 != 0:
        raise ValueError("Image dimensions must be divisible by 4")

    # Determine the size of each patch
    patch_size_rows = image_array.shape[0] // 4
    patch_size_cols = image_array.shape[1] // 4

    # Create an output image with space for the gaps and an alpha channel for transparency
    output_rows = 4 * patch_size_rows + 3 * gap
    output_cols = 4 * patch_size_cols + 3 * gap
    output_image = np.zeros((output_rows, output_cols, 4), dtype=image_array.dtype)

    # If the input image is not already 4-channel, add an alpha channel
    if image_array.shape[2] == 3:
        image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2BGRA)

    # Iterate through the image, extracting and placing each patch
    for row_idx in range(4):
        for col_idx in range(4):
            patch = image_array[
                row_idx * patch_size_rows : (row_idx + 1) * patch_size_rows,
                col_idx * patch_size_cols : (col_idx + 1) * patch_size_cols
            ]
            start_row = row_idx * (patch_size_rows + gap)
            start_col = col_idx * (patch_size_cols + gap)
            output_image[start_row : start_row + patch_size_rows, start_col : start_col + patch_size_cols] = patch

    if not gap_transparent:
        output_image = output_image[:, :, :3]

    return output_image

def stack_frames(images, pad=10, stride=20, fig_fp=None):
    h, w = 224, 224
    if type(images[0]) is np.ndarray:
        images = [Image.fromarray(img) for img in images]
    images = [img.resize((w, h)) for img in images]

    n = len(images)
    canvas_width = pad * 2 + w + (n-1) * stride
    canvas_height = pad * 2 + h + (n-1) * stride
    canvas = Image.new('RGBA', (canvas_width, canvas_height), (0, 0, 0, 0))

    for idx, img in enumerate(images):
        x, y = pad + idx * stride, pad + idx * stride
        canvas.paste(img, (x, y))

    if fig_fp is None:
        fig_fp = os.path.join(os.path.dirname(__file__), "temp", "stack_frames.png")

    canvas.resize((256, 256)).save(fig_fp)

def cat_frames(images, pad=10, gap=10, fig_fp=None):
    h, w = 224, 224
    if type(images[0]) is np.ndarray:
        images = [Image.fromarray(img) for img in images]
    images = [img.resize((w, h)) for img in images]

    n = len(images)
    canvas_width = (pad * 2) + ((w + gap) * n) -  gap
    canvas_height = (pad * 2) + h
    canvas = Image.new('RGBA', (canvas_width, canvas_height), (0, 0, 0, 0))

    for idx, img in enumerate(images):
        x, y = pad + idx * (gap + w), pad
        canvas.paste(img, (x, y))

    if fig_fp is None:
        fig_fp = os.path.join(os.path.dirname(__file__), "temp", "cat_frames.png")

    canvas.save(fig_fp)    
    
if __name__ == "__main__":
    images = [(np.random.rand(224, 224, 3) * 255).astype(np.uint8) for i in range(8)]
    stack_frames(images, fig_fp=os.path.join("temp", "stack_frames_raw.png"))
    cat_frames(images, fig_fp=os.path.join("temp", "cat_frames_raw.png"))
    output_images = [draw_patches(image) for image in images]
    stack_frames(output_images, fig_fp=os.path.join("temp", "stack_frames_pat.png"))

