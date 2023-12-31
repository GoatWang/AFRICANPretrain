import os
import copy
import torch
import numpy as np
from pathlib import Path
from torch import utils
from config import ex, config
import pytorch_lightning as pl
from datetime import datetime
from Model import VideoFrameIdenetity
from Dataset import AnimalKingdomDataset
torch.manual_seed(0)

@ex.automain
def main(_config):
    _config = copy.deepcopy(_config)

    pl.seed_everything(_config["seed"])
    dataset_valid = AnimalKingdomDataset(_config, split="val")
    _config['max_steps'] = 200000

    model = VideoFrameIdenetity(_config).to(_config['device'])
    model.set_loss_func(_config['loss'])

    test_loader = utils.data.DataLoader(dataset_valid, batch_size=_config['batch_size'], shuffle=False, num_workers=_config["data_workers"]) # bugs on MACOS

    trainer = pl.Trainer()

    # Test Clip
    trainer.test(model, dataloaders=test_loader)

    # Test AFRICAN
    model.load_ckpt_state_dict(_config['ckpt_path'])
    trainer.test(model, dataloaders=test_loader)

# WARNING - VideoFrameIdentityNetwork - No observers have been added to this run
# INFO - VideoFrameIdentityNetwork - Running command 'main'
# INFO - VideoFrameIdentityNetwork - Started
# Global seed set to 2023
# GPU available: True (cuda), used: True
# TPU available: False, using: 0 TPU cores
# IPU available: False, using: 0 IPUs
# HPU available: False, using: 0 HPUs
# You are using a CUDA device ('NVIDIA A100-SXM4-80GB') that has Tensor Cores. To properly utilize them, you should set `torch.set_float32_matmul_precision('medium' | 'high')` which will trade-off precision for performance. For more details, read https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision
# LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
# Testing DataLoader 0: 100%|█████████████████| 1524/1524 [37:19<00:00,  1.47s/it]{'valid_MulticlassAccuracy': tensor(0.3036, device='cuda:0'), 'valid_MulticlassAveragePrecision': tensor(0.2543, device='cuda:0'), 'valid_MulticlassPrecision': tensor(0.3030, device='cuda:0'), 'valid_MulticlassRecall': tensor(0.3036, device='cuda:0')}
# Testing DataLoader 0: 100%|█████████████████| 1524/1524 [37:19<00:00,  1.47s/it]
# You are using a CUDA device ('NVIDIA A100-SXM4-80GB') that has Tensor Cores. To properly utilize them, you should set `torch.set_float32_matmul_precision('medium' | 'high')` which will trade-off precision for performance. For more details, read https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision
# LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
# Testing DataLoader 0: 100%|█████████████████| 1524/1524 [36:45<00:00,  1.45s/it]{'valid_MulticlassAccuracy': tensor(0.7929, device='cuda:0'), 'valid_MulticlassAveragePrecision': tensor(0.9020, device='cuda:0'), 'valid_MulticlassPrecision': tensor(0.7928, device='cuda:0'), 'valid_MulticlassRecall': tensor(0.7929, device='cuda:0')}
# Testing DataLoader 0: 100%|█████████████████| 1524/1524 [36:45<00:00,  1.45s/it]
# INFO - VideoFrameIdentityNetwork - Completed after 1:14:40

# trained 
# {'MulticlassAccuracy': tensor(0.7695, device='cuda:0'),
#  'MulticlassAveragePrecision': tensor(0.8823, device='cuda:0'),
#  'MulticlassPrecision': tensor(0.7699, device='cuda:0'),
#  'MulticlassRecall': tensor(0.7695, device='cuda:0')}

# clip
# {'MulticlassAccuracy': tensor(0.3101, device='cuda:0'),
#  'MulticlassAveragePrecision': tensor(0.2537, device='cuda:0'),
#  'MulticlassPrecision': tensor(0.3096, device='cuda:0'),
#  'MulticlassRecall': tensor(0.3101, device='cuda:0')}

# import os
# import copy
# import torch
# import numpy as np
# from pathlib import Path
# from torch import utils
# from config import ex, config
# import pytorch_lightning as pl
# from datetime import datetime
# from Model import VideoFrameIdenetity
# from Dataset import AnimalKingdomDataset
# torch.manual_seed(0)

# def convert_to_numpy_img(frame):
#     frame_out = frame.detach().cpu().numpy().transpose(1, 2, 0)
#     minval, maxval = frame_out.min(), frame_out.max()
#     frame_out = (frame_out - minval) / (maxval - minval)
#     frame_out = (frame_out * 255).astype('uint8')
#     return frame_out

# @ex.automain
# def main(_config):
#     device = 'cuda'
#     _config = config()
#     # _config['ckpt_path'] = "/notebooks/VideoFrameIdentityNetwork/Train/ckpts/clip_cosine_infoNCE_8_uniform_augmix/20230709-215952/map_epoch=98-valid_BinaryAccuracy=0.000-valid_BinaryPrecision=0.000-valid_BinaryRecall=0.000.ckpt"
#     _config['clip_fp'] = '/notebooks/VideoFrameIdentityNetwork/Train/pretrain/ViT-L-14.pt'
#     dataset_train = AnimalKingdomDataset(_config, split="train")
#     dataset_valid = AnimalKingdomDataset(_config, split="val")

#     _config['max_steps'] = _config['max_epochs'] * len(dataset_train) // _config['batch_size']
#     model = VideoFrameIdenetity(_config).to(device)
#     model.set_loss_func(_config['loss'])
#     if _config['ckpt_path'] is not None:
#         model.load_ckpt_state_dict(_config['ckpt_path'])
#     model.eval()
#     train_loader = utils.data.DataLoader(dataset_train, batch_size=_config['batch_size'], num_workers=_config["data_workers"], shuffle=False) # TODO: DEBUG num_workers=4, maybe MACOS bug
#     valid_loader = utils.data.DataLoader(dataset_valid, batch_size=_config['batch_size'], num_workers=_config["data_workers"], shuffle=False) # TODO: DEBUG num_workers=4, maybe MACOS bug


#     save_dir = os.path.join("/", 'notebooks', "temp_Model")
#     Path(save_dir).mkdir(parents=True, exist_ok=True)

#     metric_collection = torchmetrics.MetricCollection([
#         torchmetrics.classification.MulticlassAccuracy(num_classes=_config['num_frames']),
#         torchmetrics.classification.MulticlassAveragePrecision(num_classes=_config['num_frames']),
#         torchmetrics.classification.MulticlassPrecision(num_classes=_config['num_frames']),
#         torchmetrics.classification.MulticlassRecall(num_classes=_config['num_frames'])
#     ]).to(device)