# TODO
- augumentation: 
    - Image: https://arxiv.org/pdf/2002.05709.pdf
        > 2. Method: A stochastic data augmentation module that transforms
    any given data example randomly resulting in two correlated views of the same example, denoted x˜i and x˜j ,
    which we consider as a positive pair. In this work, we
    sequentially apply three simple augmentations: random
    cropping followed by resize back to the original size, random color distortions, and random Gaussian blur. As
    shown in Section 3, the combination of random crop and
    color distortion is crucial to achieve a good performance.
    - Video: https://arxiv.org/pdf/2008.03800.pdf
    
# suspend
- Try testing it on BlueCrystol

# DONE
- Try full dataset using A100 GPU
- check output scale and logit scale of the original clip model (line 88)
- Loss infoNCE: 
    - https://github.com/RElbers/info-nce-pytorch
    - https://kevinmusgrave.github.io/pytorch-metric-learning/losses/



# Save Image Sameples
```
import os
import cv2
import sys
from config import config
sys.path.append("/notebooks/VideoFrameIdentityNetwork/Train")
from pathlib import Path
from matplotlib import pyplot as plt
from Dataset import AnimalKingdomDatasetVisualize

_config = config()
_config['data_dir'] = '/storage/AnimalKingdom/action_recognition'
dataset_valid = AnimalKingdomDatasetVisualize(_config, split="val")

video_id = 5
video_frames, video_tensor = dataset_valid[video_id]
# video_frames = convert_to_numpy_img(video_frames)

save_dir = os.path.join("/notebooks/VideoFrameIdentityNetwork/Train/temp", "samples", str(video_id).zfill(5))
raw_dir = os.path.join(save_dir, "raw")
Path(raw_dir).mkdir(exist_ok=True, parents=True)
# grid_dir = os.path.join(save_dir, "grid")

for i in range(8):
    img_fp = os.path.join(raw_dir, str(i).zfill(5)+".png")
    cv2.imwrite(img_fp, video_frames[i])
    print("file save to", img_fp)
    
%cd /notebooks/VideoFrameIdentityNetwork/Train/temp
!tar -cvzf samples.tar.gz samples
%cd /notebooks
```
