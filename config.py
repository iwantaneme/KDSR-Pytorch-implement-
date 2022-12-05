# Copyright 2021 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import random

# import numpy as np
import torch
# from torch.backends import cudnn

# Random seed to maintain reproducible results
# random.seed(0)
# torch.manual_seed(0)
# np.random.seed(0)

# Use GPU for training by default
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Turning on when the image size does not change during training can speed up training
# cudnn.benchmark = True

# Image magnification factor
upscale_factor = 4


# the number of channel(1 or 3)
num_channel = 1

# Current configuration parameter method
mode = "valid"

# Experiment name, easy to save weights and log files
exp_name = "fsrcnn_x4"

if mode == "train":
    # Dataset
    train_image_dir = f"../dataset/train_HR_NWPU"
    train_image_dir_lr = f"data/canon/train/4/LR/"
    valid_image_dir = f"../dataset/valid_HR_NWPU"

    crop_size = 120
    batch_size_train = 32
    batch_size_valid = 1
    num_workers = 4

    # Incremental training and migration training
    start_epoch = 135


    teacher_epoch = 135
    teacher_start_epoch = 1
    teacher_model = f"../ESRGAN/epochs/netG_epoch_4_{teacher_epoch}.pth"
    teacher_model = f"../ESRGAN/epochs/netG_epoch_4_{teacher_epoch}.pth"


    resume = ''
    if resume == '1':
        pretrain_G = 'epochs/netG_epoch_%d_%d.pth' % (upscale_factor, start_epoch)
        pretrain_D = 'epochs/netD_epoch_%d_%d.pth' % (upscale_factor, start_epoch)

    # Total number of epochs
    epochs = 500

    # SGD optimizer parameter
    # model_lr = 1e-3
    # model_momentum = 0.9
    # model_weight_decay = 1e-4
    # model_nesterov = False

    print_frequency = 200

if mode == "valid":
    # Test data address
    # lr_dir = f"data/Set5/LRbicx{upscale_factor}"
    sr_dir = f"results/test/test1/"
    # hr_dir = f"data/Set5/GTmod12"
    # lr_dir = f"data/canon/test/4/LR/"
    # hr_dir = f"data/canon/test/4/HR/"
    epoch = 65

    lr_dir = f"data/test/"
    hr_dir = f"../dataset/test_HR/NWPU10"

    model_path = f"netG_epoch_4_{epoch}.pth"
    student_model_path = f"student_epoch_4_{epoch}.pth"

