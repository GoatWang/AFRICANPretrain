import os
from sacred import Experiment
ex = Experiment("VideoFrameIdentityNetwork")
base_dir = os.path.dirname(__file__)

@ex.config
def config():
    # basic
    name = "VideoFrameIdentityNetwork" # <clip/init>_<BCE/FOCAL>_<bs>
    seed = 2023
    device = 'cpu' # cuda

    # for save
    log_dir = os.path.abspath(os.path.join(base_dir, "logs"))
    model_dir = os.path.abspath(os.path.join(base_dir, "ckpts"))
    data_dir = os.path.join(base_dir, "..", "..", "data", "AnimalKingdom", "action_recognition")

    # for training
    num_frames = 8
    video_sampling = 'rand' # 'sequence_rand'
    batch_size = 16
    max_epochs = 100
    lr = 0.0001
    optimizer = "adamw" # adam or adamw
    decay_power = "no_decay" # no_decay, poly, cosine, linear
    warmup_steps = 10000 # https://chat.openai.com/share/ff341d8f-77dc-4a57-bc3b-a47210fe6b2e
    end_lr = 0.0 # for poly decay
    poly_decay_power = 1 # for poly decay

    # for dataset
    version = None
    data_workers = 12
    functional_test_size = None
    limit_train_batches = 0.2
    limit_valid_batches = 0.2
    clip_fp = None # os.path.join(base_dir, "pretrain", "ViT-L-14.pt")

    # for model
    loss = "BCE" # "BCE", "FOCAL", "LDAM", "EQL"
    train_laryers = "all" # all, vision_proj, None
    ckpt_path = None

    # temp_dir 
    temp_dir = os.path.join(base_dir, "temp")
    