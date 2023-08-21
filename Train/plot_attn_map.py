import os
import cv2
import copy
import torch
import numpy as np
from pathlib import Path
from torch import utils
from config import ex, config
import pytorch_lightning as pl
from datetime import datetime
from matplotlib import pyplot as plt
from Model import VideoFrameIdenetity
from Dataset import AnimalKingdomDatasetVisualize

def forward_for_visual(x, model_visual):
    """
    # video_tensors, labels_onehot = next(iter(valid_loader))
    # video_tensor1, video_tensor2, labels_onehot = video_tensors[0].to(_config['device']), video_tensors[1].to(_config['device']), labels_onehot.to(_config['device'])

    # x1 = video_tensor1[:, 0] # 3, 224, 224
    # x2 = video_tensor1[:, 1] # 3, 224, 224
    # print(x1.shape)
    # print(x2.shape)

    # pooled, tokens, attn_output_weights_layers = forward_for_visual(model_visual, x1)
    # print("len(attn_output_weights_layers)", len(attn_output_weights_layers))
    # print("attn_output_weights_layers[0].shape", attn_output_weights_layers[0].shape)
    """
    x = model_visual.conv1(x)  # shape = [*, width, grid, grid]
    x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
    x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

    # class embeddings and positional embeddings
    x = torch.cat(
        [model_visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
         x], dim=1)  # shape = [*, grid ** 2 + 1, width]
    x = x + model_visual.positional_embedding.to(x.dtype)

    # a patch_dropout of 0. would mean it is disabled and this function would do nothing but return what was passed in
    x = model_visual.patch_dropout(x)
    x = model_visual.ln_pre(x)

    x = x.permute(1, 0, 2)  # NLD -> LND
    
    model_transformer = model_visual.transformer
    attn_output_weights_layers = []
    for resblock in model_transformer.resblocks:
        # resblock.attention(q_x=resblock.ln_1(x), k_x=None, v_x=None, attn_mask=None)
        # self.attn(x, k_x, v_x, need_weights=False, attn_mask=attn_mask)[0]
        q_x = resblock.ln_1(x)
        x_attn, attn_output_weights = resblock.attn(q_x, q_x, q_x, need_weights=True, average_attn_weights=False, attn_mask=None)
        x = q_x + resblock.ls_1(x_attn)
        x = x + resblock.ls_2(resblock.mlp(resblock.ln_2(x)))
        attn_output_weights_layers.append(attn_output_weights)
        
    x = x.permute(1, 0, 2)  # LND -> NLD
    pooled, tokens = model_visual._global_pool(x)
    pooled = model_visual.ln_post(pooled)
    pooled = pooled @ model_visual.proj
    return pooled, tokens, attn_output_weights_layers

def get_attention_map(img, img_tensor, model_visual, get_mask=False, device='cpu'):
    with torch.no_grad():
        pooled, tokens, attn_output_weights_layers = forward_for_visual(img_tensor.unsqueeze(0), model_visual)
    att_mat = torch.stack(attn_output_weights_layers).squeeze(1)

    # Average the attention weights across all heads.
    att_mat = torch.mean(att_mat, dim=1)

    # To account for residual connections, we add an identity matrix to the
    # attention matrix and re-normalize the weights.
    residual_att = torch.eye(att_mat.size(1)).to(device)
    aug_att_mat = att_mat + residual_att
    aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)

    # Recursively multiply the weight matrices
    joint_attentions = torch.zeros(aug_att_mat.size()).to(device)
    joint_attentions[0] = aug_att_mat[0]
    
    for n in range(1, aug_att_mat.size(0)):
        joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n-1])
        
    v = joint_attentions[-1]
    grid_size = int(np.sqrt(aug_att_mat.size(-1)))
    mask = v[0, 1:].reshape(grid_size, grid_size).detach().cpu().numpy()
    
    mask = cv2.resize(mask / mask.max(), img.shape[:2])
    heatmap = cv2.applyColorMap((mask*255).astype(np.uint8), cv2.COLORMAP_JET)
    result = cv2.addWeighted(img, 0.7, heatmap, 0.3, 0.0)
    
    return mask, heatmap, result

# def plot_attention_map(original_img, att_map_clip, att_map_africa):
#     fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 4))
#     ax1.set_title('Original')
#     ax2.set_title('Clip Heatmap')
#     ax3.set_title('Africa Heatmap')
#     ax1.imshow(original_img)
#     ax2.imshow(att_map_clip)
#     ax3.imshow(att_map_africa)
#     plt.show()    

def turn_off_axis_ticks(ax):
    # ax.axis('off')  # Turn off the axis lines
    ax.set_xticks([])  # Turn off the x-axis ticks
    ax.set_yticks([])  # Turn off the y-axis ticks
    
def plot_attention_map_v2(video_frames, video_tensor, model_clip, model_africa, device, fig_fp=None):
    n_rows, n_cols = 3, 8
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5))
    
    for ci in range(n_cols):
        mask, heatmap, result_clip = get_attention_map(video_frames[ci], video_tensor[ci], model_clip.image_encoder, get_mask=True, device=device)
        mask, heatmap, result_africa = get_attention_map(video_frames[ci], video_tensor[ci], model_africa.image_encoder, get_mask=True, device=device)
        axes[0][ci].imshow(video_frames[ci])
        axes[1][ci].imshow(result_clip)
        axes[2][ci].imshow(result_africa)

        axes[0][ci].set_title(f"frame{ci+1}")
        for ri in range(n_rows):
            turn_off_axis_ticks(axes[ri][ci])

    axes[0][0].set_ylabel('Raw')
    axes[1][0].set_ylabel('Clip')
    axes[2][0].set_ylabel('African')
    plt.suptitle("Attention Heatmap")

    if fig_fp:
        plt.savefig(fig_fp)
        plt.close()
    else:
        plt.show()

def convert_to_numpy_img(frame):
    frame_out = frame.detach().cpu().numpy().transpose(1, 2, 0)
    minval, maxval = frame_out.min(), frame_out.max()
    frame_out = (frame_out - minval) / (maxval - minval)
    frame_out = (frame_out * 255).astype('uint8')
    return frame_out

@ex.automain
def main(_config):
    _config = copy.deepcopy(_config)

    _config['batch_size'] = 1
    _config['device'] = 'cuda'
    _config['data_dir'] = '/storage/AnimalKingdom/action_recognition'
    _config['ckpt_path'] = "/notebooks/AnimalKingdomCLIP/Train/weights/clip_nodecay_infoNCE_8_rand_augmix_000030_epoch30.ckpt"
    _config['clip_fp'] = '/notebooks/VideoFrameIdentityNetwork/Train/pretrain/ViT-L-14.pt'
    Path(_config['attn_map_dir']).mkdir(exist_ok=True, parents=True)
    dataset_valid = AnimalKingdomDatasetVisualize(_config, split="val", mode="attnmap")

    _config['max_steps'] = _config['max_epochs'] * len(dataset_valid) // _config['batch_size']
    model_clip = VideoFrameIdenetity(_config).to(_config['device'])
    model_clip.eval()

    model_africa = VideoFrameIdenetity(_config).to(_config['device'])
    model_africa.load_ckpt_state_dict(_config['ckpt_path'])
    model_africa.eval()
    print("model load success")

    for i in [30, 60, 80, 85, 90, 140, 147, 153]:
        video_frames, video_tensor = dataset_valid[i]
        video_tensor = video_tensor.to(_config['device'])
        fig_fp = os.path.join(_config['temp_dir'], "attn_map", str(i).zfill(5) + ".png")
        plot_attention_map_v2(video_frames, video_tensor, model_clip, model_africa, _config['device'], fig_fp)
        print("file saved to ", fig_fp)
        

    # i = 161 # 
    # for i in range(i, i+5):
    #     print(i)
    #     video_frames, video_tensor = dataset_valid[i]
    #     video_tensor = video_tensor.to(_config['device'])
    #     print(video_frames.shape)
    #     print(video_tensor.shape)

    #     for i in range(8):
    #         mask, heatmap, result_clip = get_attention_map(video_frames[i], video_tensor[i], model_clip.image_encoder, get_mask=True, device=_config['device'])
    #         mask, heatmap, result_africa = get_attention_map(video_frames[i], video_tensor[i], model_africa.image_encoder, get_mask=True, device=_config['device'])
    #         plot_attention_map(video_frames[i], result_clip, result_africa)

    #         # TODO: savefig
    #         # TODO: save as gif

