import os
import glob
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from VideoReader import read_frames_decord
from Transform import VideoTransformTorch, VideoTransformSelfdefined, VideoTransformVisualize, video_aug

class AnimalKingdomDataset(torch.utils.data.Dataset):
    def __init__(self, config, split=""):
        assert split in ["train", "val"], "split must be train or val"
        self.split = split
        self.metadata = None
        self.ans_lab_dict = dict()
        self.device = config['device'] 
        self.data_dir = config['data_dir']
        self.num_frames = config['num_frames']
        self.video_sampling = config['video_sampling']
        self.functional_test_size = config['functional_test_size']

        self.video_transform = VideoTransformTorch(mode='train')  # NOTICE: all split shuould be augmentated
        # self.video_transform = VideoTransformSelfdefined(mode='train')  # NOTICE: all split shuould be augmentated
        self.video_aug = video_aug
        self.load_metadata()

    def process_annotation(self, csv_fp, video_fps):
        video_id_mapping = {os.path.basename(fp).replace(".mp4", ""):fp for fp in video_fps}

        # group into one video per line
        df = pd.read_csv(csv_fp, sep=' ')
        df_out1 = df.groupby("original_vido_id").first().reset_index()
        df_out2 = df.groupby("original_vido_id")['path'].apply(len).reset_index()
        df_out2.columns = ['original_vido_id', 'count']
        df_out = pd.merge(df_out1.drop('path', axis=1), df_out2, on='original_vido_id')

        # add features
        df_out['video_fp'] = df_out['original_vido_id'].apply(video_id_mapping.get)
        df_out['labels'] = df_out['labels'].apply(lambda x: [int(l) for l in x.split(",")])
        df_out = df_out[df_out['video_fp'].notnull()].reset_index(drop=True)
        end_idx = self.functional_test_size if self.functional_test_size else len(df_out)
        return df_out['video_fp'].loc[:end_idx].tolist(), df_out['labels'].loc[:end_idx].tolist()

    def load_metadata(self):
        self.df_action = pd.read_excel(os.path.join(self.data_dir, 'annotation', 'df_action.xlsx'))
        # self.df_metadata = pd.read_excel(os.path.join(self.data_dir, 'AR_metadata.xlsx'))
        video_fps = glob.glob(os.path.join(self.data_dir, 'dataset', 'video', "*.mp4"))
        split_files = {
            'train': os.path.join(self.data_dir, "annotation", "train.csv"),
            'val': os.path.join(self.data_dir, "annotation", "val.csv")
        }
        target_split_fp = split_files[self.split]
        self.video_fps, self.labels = self.process_annotation(target_split_fp, video_fps)

    def __getitem__(self, index):
        ret = None
        video_fp = self.video_fps[index]
        video_tensor, frame_idxs, vlen = read_frames_decord(video_fp, num_frames=self.num_frames, sample=self.video_sampling)
        video_tensor1 = self.video_aug(video_tensor, self.video_transform)
        video_tensor2 = self.video_aug(video_tensor, self.video_transform)
        labels = torch.eye(self.num_frames)
        if len(frame_idxs) < self.num_frames:
            pad_n_frames = self.num_frames - len(frame_idxs)
            labels[-pad_n_frames:, -pad_n_frames:] = 1
        return (video_tensor1, video_tensor2), labels
    
    def __len__(self):
        return len(self.video_fps)

class AnimalKingdomDatasetVisualize(AnimalKingdomDataset):
    def __init__(self, config, split="val", mode='attnmap'):
        """
        mode: {"simmat", "attnmap"}
        """
        super().__init__(config, split)
        self.video_transform = VideoTransformTorch(mode='train')  # NOTICE: all split shuould be augmentated
        self.video_transform_norm = VideoTransformVisualize()

        self.mode = mode

    def __getitem__(self, index):
        video_fp = self.video_fps[index]
        video_frames_raw, frame_idxs, vlen = read_frames_decord(video_fp, num_frames=self.num_frames, sample="uniform")
        if self.mode == "attnmap":
            video_frames = self.video_aug(video_frames_raw, self.video_transform).detach().cpu().numpy()
            video_frames = (video_frames.transpose(0, 2, 3, 1) * 255).astype("uint8")
            video_tensor = self.video_aug(video_frames_raw, self.video_transform_norm)
            return video_frames, video_tensor
        
        elif self.mode == "simmat":
            video_frames_raw_out = (video_frames_raw.detach().cpu().numpy().transpose(0, 2, 3, 1) * 255).astype("uint8")
            video_frames1 = self.video_aug(video_frames_raw, self.video_transform).detach().cpu().numpy()
            video_frames1 = (video_frames1.transpose(0, 2, 3, 1) * 255).astype("uint8")
            video_frames2 = self.video_aug(video_frames_raw, self.video_transform).detach().cpu().numpy()
            video_frames2 = (video_frames2.transpose(0, 2, 3, 1) * 255).astype("uint8")
            return video_fp, video_frames_raw_out, video_frames1, video_frames2
    
    def __len__(self):
        return len(self.video_fps)
    


if __name__  == "__main__":
    from config import config
    _config = config()
    # # ===============run all data test
    # from torch import utils
    # dataset_train = AnimalKingdomDataset(_config, split="train")
    # dataset_valid = AnimalKingdomDataset(_config, split="val")
    # train_loader = utils.data.DataLoader(dataset_train, batch_size=4, shuffle=False, num_workers=_config['data_workers']) # TODO: DEBUG num_workers=4, maybe MACOS bug
    # valid_loader = utils.data.DataLoader(dataset_valid, batch_size=4, shuffle=False, num_workers=_config['data_workers']) # TODO: DEBUG num_workers=4, maybe MACOS bug
    # print(len(train_loader))
    # for batch_idx, (video_tensor, labels_onehot) in enumerate(train_loader):
    #   print(batch_idx, "success")

    # print(len(valid_loader))
    # for batch_idx, (video_tensor, labels_onehot) in enumerate(valid_loader):
    #   print(batch_idx, "success")
    # # ===============

    dataset = AnimalKingdomDataset(_config, split="train")
    idx = [i for i, v in enumerate(dataset.video_fps) if "AAOZGQFB" in v][0] # short video
    (video_tensor1, video_tensor2), labels = dataset[idx]
    print(labels)
    print(video_tensor1.shape, video_tensor2.shape, labels.shape)


    import cv2
    from matplotlib import pyplot as plt
    def frame2img(frame):
        frame_out = frame.numpy().transpose(1, 2, 0)
        minval, maxval = frame_out.min(), frame_out.max()
        if maxval - minval > 0:
            frame_out = (frame_out - minval) / (maxval - minval)
        frame_out = (frame_out * 255).astype('uint8')
        return frame_out[:, :, ::-1]
    
    save_dir = os.path.join(os.path.dirname(__file__), "temp", "Dataset")
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    for frame_idx, (frame1, frame2) in enumerate(zip(video_tensor1, video_tensor2)):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        ax1.imshow(frame2img(frame1))
        ax2.imshow(frame2img(frame2))
        plt.savefig(os.path.join(save_dir, f"{frame_idx}.jpg"))





