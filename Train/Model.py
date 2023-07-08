import os
import math
import json
import numpy as np
from copy import deepcopy

import torch
import torchmetrics
from Loss import get_loss_func
import pytorch_lightning as pl
from open_clip import CLIPVisionCfg, _build_vision_tower, create_model_and_transforms
from transformers import (
    get_polynomial_decay_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)

class VideoFrameIdenetity(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.lr = config["lr"]
        self.optimizer = config["optimizer"]

        self.decay_power = config['decay_power']
        self.warmup_steps = config['warmup_steps']
        self.max_steps = config['max_steps']
        self.end_lr = config['end_lr']
        self.poly_decay_power = config['poly_decay_power']
        self.num_classes = config['num_frames']

        config_fp = os.path.join(os.path.dirname(__file__), "open_clip/model_configs/ViT-L-14.json")
        with open(config_fp, 'r') as f:
            model_config = json.load(f)

        self.image_encoder = _build_vision_tower(model_config['embed_dim'], model_config['vision_cfg'])
        if config['clip_fp'] is not None:
            clip_model = torch.jit.load(config['clip_fp'])
            self.image_encoder.load_state_dict(clip_model.visual.state_dict())        

        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        metric_collection = torchmetrics.MetricCollection([
            torchmetrics.classification.MulticlassAccuracy(num_classes=self.num_classes),
            torchmetrics.classification.MulticlassAveragePrecision(num_classes=self.num_classes),
            torchmetrics.classification.MulticlassPrecision(num_classes=self.num_classes),
            torchmetrics.classification.MulticlassRecall(num_classes=self.num_classes)
        ])
        self.train_metrics = metric_collection.clone(prefix='train_')
        self.valid_metrics = metric_collection.clone(prefix='valid_')

        # train_laryers
        # all: freeze nothing
        # None: freeze all parameters
        # vision_proj: freeze_image_encoder_except_proj
        if config['train_laryers'] is None:
            self.freeze_image_encoder()
        if config['train_laryers'] == "vision_proj":
            self.freeze_image_encoder_except_proj()

    def load_ckpt_state_dict(self, ckpt_fp):
        ckpt = torch.load(ckpt_fp, map_location="cpu")
        state_dict = ckpt["state_dict"]
        self.load_state_dict(state_dict, strict=False)

    def set_loss_func(self, loss_name): # BCE recommend
        self.loss_func = get_loss_func(loss_name)

    def freeze_image_encoder(self):
        for n, p in self.named_parameters():
            p.requires_grad = False

    def freeze_image_encoder_except_proj(self):
        self.freeze_image_encoder()
        self.proj.requires_grad = True
    
    def forward_single_video(self, video_tensor):
        B, F, C, H, W = video_tensor.shape
        video_tensor = video_tensor.contiguous().view(B*F, C, H, W)
        frame_feat = self.image_encoder(video_tensor).view(B, F, -1) # B, F, 768
        return frame_feat

    def forward(self, batch, mode="video"):
        (video_tensor1, video_tensor2), labels_onehot = batch
        frame_feat1 = self.forward_single_video(video_tensor1)
        frame_feat2 = self.forward_single_video(video_tensor2)
        return (frame_feat1, frame_feat2)
        
    # def cal_similiarity(self, frame_feat1, frame_feat2):
    #     # (8, 768) @ (768, 8) -> (8, 8)
    #     frame_similiarity_logits = frame_feat1 @ frame_feat2.t() * self.logit_scale.exp()
    #     return frame_similiarity_logits
    
    def cal_similarity_parallel(self, frame_feat1, frame_feat2):
        # (B, 8, 768) @ (B, 768, 8) -> (B, 8, 8)
        frame_similiarity_logits = torch.bmm(frame_feat1, frame_feat2.transpose(1, 2)) * self.logit_scale.exp()
        return frame_similiarity_logits.view(-1, frame_similiarity_logits.shape[-1])

    def training_step(self, batch, batch_idx):
        _, labels_onehot = batch
        frame_feat1, frame_feat2 = self(batch)

        # frame_logits = torch.stack([self.cal_similiarity(frame_feat1[b], frame_feat2[b]) for b in range(B)]).view(-1, labels_onehot.shape[-1])
        frame_logits = self.cal_similarity_parallel(frame_feat1, frame_feat2)
        ground_truth = labels_onehot.to(frame_logits.device).view(-1, labels_onehot.shape[-1])
        loss = self.loss_func(frame_logits, ground_truth.type(torch.float32))
        self.train_metrics.update(torch.softmax(frame_logits, dim=1), torch.argmax(ground_truth, dim=1))
        self.log("train_loss", loss)
        return loss

    def on_train_epoch_end(self):
        _train_metrics = self.train_metrics.compute()
        self.log_dict(_train_metrics)
        self.train_metrics.reset()

    def validation_step(self, batch, batch_idx):
        _, labels_onehot = batch
        frame_feat1, frame_feat2 = self(batch)

        # frame_logits = torch.stack([self.cal_similiarity(frame_feat1[b], frame_feat2[b]) for b in range(B)]).view(-1, labels_onehot.shape[-1])
        frame_logits = self.cal_similarity_parallel(frame_feat1, frame_feat2)
        ground_truth = labels_onehot.to(frame_logits.device).view(-1, labels_onehot.shape[-1])
        loss = self.loss_func(frame_logits, ground_truth.type(torch.float32))
        self.valid_metrics.update(torch.softmax(frame_logits, dim=1), torch.argmax(ground_truth, dim=1))
        self.log("valid_loss", loss)

    def on_validation_epoch_end(self):
        _valid_metrics = self.valid_metrics.compute()
        self.log_dict(_valid_metrics)
        self.valid_metrics.reset()

    def configure_optimizers(self):
        if self.optimizer == 'adam':
            optimizer = torch.optim.Adam([p for p in self.parameters() if p.requires_grad], lr=self.lr)
        elif self.optimizer == 'adamw':
            optimizer = torch.optim.AdamW([p for p in self.parameters() if p.requires_grad], lr=self.lr, eps=1e-6, betas=(0.9, 0.98))
        else:
            assert False, f"Unknown optimizer: {optimizer}"

        if self.decay_power == "no_decay":
            return optimizer
        else:
            if self.decay_power == "cosine":
                scheduler = get_cosine_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=self.warmup_steps,
                    num_training_steps=self.max_steps,
                )
            elif self.decay_power == "poly":
                scheduler = get_polynomial_decay_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=self.warmup_steps,
                    num_training_steps=self.max_steps,
                    lr_end=self.end_lr,
                    power=self.poly_decay_power,
                )
            sched = {"scheduler": scheduler, "interval": "step"}

            return ([optimizer], [sched])    
    
if __name__ == "__main__":    
    import numpy as np
    from torch import utils
    from pathlib import Path
    from config import config
    from Dataset import AnimalKingdomDataset
    save_dir = os.path.join(os.path.dirname(__file__), "temp", "Model")
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    device = 'cpu'
    _config = config()
    dataset_train = AnimalKingdomDataset(_config, split="train")
    dataset_valid = AnimalKingdomDataset(_config, split="val")

    _config['max_steps'] = _config['max_epochs'] * len(dataset_train) // _config['batch_size']
    model = VideoFrameIdenetity(_config)
    model.set_loss_func(_config['loss'])
    # ckpt_fp = os.path.join(os.path.dirname(__file__), "weights", "epoch=2-step=9003.ckpt")
    # if os.path.exists(ckpt_fp):
    #     model.load_ckpt_state_dict(ckpt_fp)

    train_loader = utils.data.DataLoader(dataset_train, batch_size=1, shuffle=True) # TODO: DEBUG num_workers=4, maybe MACOS bug
    valid_loader = utils.data.DataLoader(dataset_valid, batch_size=1, shuffle=False) # TODO: DEBUG num_workers=4, maybe MACOS bug

    # # test inference
    # for batch_idx, ((video_tensor1, video_tensor2), labels_onehot) in enumerate(train_loader):
    #     video_tensor1, video_tensor2, labels_onehot = video_tensor1.to(device), video_tensor2.to(device), labels_onehot.to(device)
    #     frame_feat1, frame_feat2 = model(((video_tensor1, video_tensor2), labels_onehot))
    #     frame_feat1, frame_feat2 = frame_feat1.cpu().detach().numpy(), frame_feat2.cpu().detach().numpy()
    #     print("frame_feat1.shape", frame_feat1.shape)
    #     print("frame_feat2.shape", frame_feat2.shape)
    #     np.save(os.path.join(save_dir, "frame_feat1.npy"), frame_feat1)
    #     np.save(os.path.join(save_dir, "frame_feat2.npy"), frame_feat2)
    #     break
    # # frame_feat1 = np.load(os.path.join(save_dir, "frame_feat1.npy"))
    # # frame_feat2 = np.load(os.path.join(save_dir, "frame_feat2.npy"))
    

    # test otptimizer
    optimizer = model.configure_optimizers()

    # test forward and train
    for batch_idx, ((video_tensor1, video_tensor2), labels_onehot) in enumerate(train_loader):
        video_tensor1, video_tensor2, labels_onehot = video_tensor1.to(device), video_tensor2.to(device), labels_onehot.to(device)
        loss = model.training_step(((video_tensor1, video_tensor2), labels_onehot), batch_idx)
        print(loss)
        break

    for batch_idx, ((video_tensor1, video_tensor2), labels_onehot) in enumerate(valid_loader):
        video_tensor1, video_tensor2, labels_onehot = video_tensor1.to(device), video_tensor2.to(device), labels_onehot.to(device)
        model.validation_step(((video_tensor1, video_tensor2), labels_onehot), batch_idx)
        break


