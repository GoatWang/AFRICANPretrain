import os
import cv2
import copy
import numpy as np
from pathlib import Path
from config import ex, config
from PIL import Image, ImageDraw, ImageFont
from Dataset import AnimalKingdomDatasetVisualize
from plot_patches import stack_frames, cat_frames, draw_patches


def write_font(draw, text, x, y, anchor="mm", font_size=80, color='white'):
    """
    x, y: center of the text
    anchor: https://pillow.readthedocs.io/en/stable/handbook/text-anchors.html#text-anchors
    """
    # font = ImageFont.truetype("Arial Unicode.ttf", size=font_size) # For Mac
    font = ImageFont.truetype(os.path.join(os.path.dirname(__file__), "fonts/arimo/Arimo-Regular.ttf"), size=font_size) # For Mac
    draw.text((x, y), text, fill=color, font=font, anchor=anchor)

def draw_3points(h, w, color='white'):
        img = Image.new('RGBA', (w, h), (0, 0, 0, 0))
        img_draw = ImageDraw.Draw(img)
        write_font(img_draw, "...", w//2, h//2, anchor="ms", color=color)
        return img
    
def plot_contrastive_learning_structure(images_ci, images_ri, color='black', pad=30, gap=30, fig_fp=None):
    h, w = 224, 224
    n_rows, n_cols = len(images_ri), len(images_ci)

    images_ci = [img.resize((w, h)) for img in images_ci]
    images_ri = [img.resize((w, h)) for img in images_ri]

    # Define the coordinates (x, y) for the top-left corner of each image
    coords_ci = [(pad + (h+gap) * (i+1), pad) for i in range(n_cols)]
    coords_ri = [(pad, pad + (w+gap) * (i+1)) for i in range(n_rows)]

    # Create a blank canvas with a transparent background
    canvas_width = (w + gap) * (len(coords_ci)+1) + pad*2
    canvas_height = (h + gap) * (len(coords_ri)+1) + pad*2
    canvas = Image.new('RGBA', (canvas_width, canvas_height), (0, 0, 0, 0))

    # Paste each image onto the canvas at the specified coordinates
    for idx, (img, coord) in enumerate(zip(images_ci, coords_ci)):
        if idx != len(images_ci) - 2:
            canvas.paste(img, coord)
        else:
            img = draw_3points(h, w, color=color)
            canvas.paste(img, coord)

    for idx, (img, coord) in enumerate(zip(images_ri, coords_ri)):
        if idx != len(images_ri) - 2:
            canvas.paste(img, coord)
        else:
            img = draw_3points(h, w, color=color)
            canvas.paste(img.rotate(90), coord)

    draw = ImageDraw.Draw(canvas)

    # vertical lines
    wg, hg = w+gap, h+gap
    vlines = [(pad+((i+1)*wg) - wg - gap/2, 
            pad + hg - gap/2, 
            pad+((i+1)*wg)-wg-gap/2,
            pad + hg * (n_cols+1) - gap/2) for i in range(1, n_cols+2)]
    hlines = [(pad + wg - gap/2, 
            pad + ((i+1)*hg) - hg - gap/2, 
            pad + wg*(n_rows+1) - gap/2, 
            pad + ((i+1)*hg) - hg - gap/2) for i in range(1, n_rows+2)]

    for line in vlines+hlines:
        draw.line(line, fill=color, width=7)

    for ri in range(1, n_rows+1):
        for ci in range(1, n_cols+1):
            x, y = pad + (wg * ci) + w/2, pad + (hg * ri) + h/2
            if (ri != (n_rows - 1)) and (ci != (n_cols - 1)) or (ri == (n_rows)) or (ci == (n_cols)):
                text = "1" if ri == ci else "0"
                write_font(draw, text, x, y, color=color)
            elif (ci == (n_cols - 1)) and (ri == (n_rows - 1)):
                img = draw_3points(h, w, color=color)
                canvas.paste(img.rotate(135), (int(x-w//2), int(y-h//2)))
            elif (ri == (n_rows - 1)):
                img = draw_3points(h, w, color=color)
                canvas.paste(img.rotate(90), (int(x-w//2), int(y-h//2)))
            elif (ci == (n_cols - 1)):
                img = draw_3points(h, w, color=color)
                canvas.paste(img, (int(x-w//2), int(y-h//2)))

    # Save the final image as a PNG file
    if fig_fp is None:
        fig_fp = os.path.join(os.path.dirname(__file__), "temp", 'SimilarityMatrix.png')
    
    canvas.resize((716, 716)).save(fig_fp)

# if __name__ == "__main__":
#     # Define your 8 images as numpy ndarrays
#     n_rows, n_cols = 8, 8
#     images_ci = [Image.fromarray((np.random.rand(224, 224, 3) * 255).astype(np.uint8))] * n_cols
#     images_ri = [Image.fromarray((np.random.rand(224, 224, 3) * 255).astype(np.uint8))] * n_rows
#     plot_contrastive_learning_structure(np.array(images_ci)[[1, 2, 3, 4, 5, 7]], np.array(images_ri)[[1, 2, 3, 4, 5, 7]])

@ex.automain
def main(_config):
    _config = copy.deepcopy(_config)

    dataset_valid = AnimalKingdomDatasetVisualize(_config, split="val", mode="simmat")
    _config['max_steps'] = 200000

    save_dir_african = os.path.join(_config['temp_dir'], "AFRICANSimilarityMatrix")
    save_dir_frame_concat = os.path.join(_config['temp_dir'], "FrameConcat")
    save_dir_frame_stack = os.path.join(_config['temp_dir'], "FrameStack")
    save_dir_single = os.path.join(_config['temp_dir'], "Single")

    Path(save_dir_african).mkdir(exist_ok=True, parents=True)
    Path(save_dir_frame_concat).mkdir(exist_ok=True, parents=True)
    Path(save_dir_frame_stack).mkdir(exist_ok=True, parents=True)
    Path(save_dir_single).mkdir(exist_ok=True, parents=True)

    for idx in np.random.choice(range(len(dataset_valid)), 30):
        video_fp, video_frames_raw, video_frames1, video_frames2 = dataset_valid[idx]
        fig_fn = os.path.basename(video_fp).split('.')[0]

        fig_fp_raw = os.path.join(save_dir_frame_concat, fig_fn+".png")
        fig_fp_stack_raw = os.path.join(save_dir_frame_stack, fig_fn+".png")
        fig_fp_stack_patch = os.path.join(save_dir_frame_stack, fig_fn+"_patched.png")
        fig_fp_african = os.path.join(save_dir_african, fig_fn+"_african.png")

        cat_frames(video_frames_raw, fig_fp=fig_fp_raw)
        stack_frames(video_frames_raw, fig_fp=fig_fp_stack_raw)
        stack_frames([draw_patches(f) for f in video_frames_raw], fig_fp=fig_fp_stack_patch)

        video_frames1 = [Image.fromarray(video_frames1[i]) for i in [1, 2, 3, 4, 5, 7]]
        video_frames2 = [Image.fromarray(video_frames2[i]) for i in [1, 2, 3, 4, 5, 7]]
        plot_contrastive_learning_structure(video_frames1, video_frames2, fig_fp=fig_fp_african)

        for idx, frame in enumerate(video_frames_raw):
            fig_fp_raw = os.path.join(save_dir_single, fig_fn+"_raw_%02i.png"%idx)
            cv2.imwrite(fig_fp_raw, frame)
            fig_fp_pat = os.path.join(save_dir_single, fig_fn+"_pat_%02i.png"%idx)
            cv2.imwrite(fig_fp_pat, draw_patches(frame))

        print("file saved to ", fig_fp_raw)
        print("file saved to ", fig_fp_stack_raw)
        print("file saved to ", fig_fp_stack_patch)
        print("file saved to ", fig_fp_african)
        print("==================")



# %cd /notebooks
# !python3 VideoFrameIdentityNetwork/Train/plot_similarity_matrix.py with 'batch_size=1' 'device="cuda"' \
# 'data_dir="/storage/AnimalKingdom/action_recognition"' \
# 'clip_fp="/notebooks/VideoFrameIdentityNetwork/Train/pretrain/ViT-L-14.pt"'

# %cd /notebooks/VideoFrameIdentityNetwork/Train
# !tar -cvzf temp.tar.gz temp
# %cd /notebooks

