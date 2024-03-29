import argparse
import os
import pandas as pd
from tqdm import tqdm
import numpy as np
import torch
import torch.optim as optim
import torch.utils.data
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from math import log10
import pytorch_ssim
from loss import *
from data_utils import TrainDatasetFromFolder, ValDatasetFromFolder, display_transform
from model.ESRGAN import ESRGAN as Generator
import math
from feature_transformation import spatial_similarity, channel_similarity, batch_similarity, FSP, AT

import config

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_teacher():
    print("加载教师模型 ====================================>")
    teacher = Generator(3, 3, scale_factor=config.upscale_factor).to(device)
    checkpointT = torch.load(config.teacher_model, map_location=config.device)
    netT_state_dict = teacher.state_dict()
    newT_state_dict = {k: v for k, v in checkpointT["state_dict"].items() if k in netT_state_dict}
    netT_state_dict.update(newT_state_dict)
    teacher.load_state_dict(netT_state_dict)
    # teacher.load_state_dict(torch.load(config.teacher_model))
    for p in teacher.parameters():
        p.requires_grad = False
    return teacher


def create_student_model():
    print("构建学生模型 ===================================>")
    student = Generator(3, 3, scale_factor=config.upscale_factor, n_basic_block=10).to(device)
    checkpointS = torch.load(config.teacher_model, map_location=config.device)
    netS_state_dict = teacher.state_dict()
    newS_state_dict = {k: v for k, v in checkpointS["state_dict"].items() if k in netS_state_dict}
    netS_state_dict.update(newS_state_dict)
    student.load_state_dict(netS_state_dict, False)
    # student.load_state_dict(torch.load(config.teacher_model), False)
    return student


def prepare_criterion(kdtype):
    # 准备loss函数
    criterion = Loss(kdtype).to(device)
    return criterion


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='训练超分模型')
    # parser.add_argument('--crop_size', default=120, type=int, help='训练图片裁剪尺寸')
    # parser.add_argument('--upscale_factor', default=4, type=int, choices=[2, 4, 8],
    #                     help='超分倍数')
    # parser.add_argument('--num_epochs', default=100, type=int, help='训练迭代次数')
    #
    # opt = parser.parse_args()

    CROP_SIZE = config.crop_size
    UPSCALE_FACTOR = config.upscale_factor
    NUM_EPOCHS = config.epochs
    # 训练集（urban100和Sun-Hays80组成）和验证集Set14组成
    train_set = TrainDatasetFromFolder(config.train_image_dir, crop_size=CROP_SIZE, upscale_factor=UPSCALE_FACTOR)
    val_set = ValDatasetFromFolder(config.valid_image_dir, upscale_factor=UPSCALE_FACTOR)
    train_loader = DataLoader(dataset=train_set, num_workers=config.num_workers, batch_size=config.batch_size_train,
                              shuffle=True)
    test_loader = DataLoader(dataset=val_set, num_workers=config.num_workers, batch_size=config.batch_size_valid,
                             shuffle=False)

    # 特征蒸馏类型
    feature_distilation_type = '1*SA'

    teacher = load_teacher()
    student = create_student_model()
    criterion = prepare_criterion(feature_distilation_type)
    optimizer = optim.Adam(student.parameters(), lr=1e-4, betas=(0.9, 0.999), weight_decay=0)

    results = {'loss': [], 'psnr': [], 'ssim': []}

    if config.resume:
        print("Check whether the pretrained model is restored...")
        # Load checkpoint model
        checkpointG = torch.load(config.pretrain_G, map_location=config.device)
        checkpointD = torch.load(config.pretrain_D, map_location=config.device)
        # Restore the parameters in the training node to this point
        config.start_epoch = checkpointG["epoch"] + 1
        # Load checkpoint state dict. Extract the fitted model weights
        netG_state_dict = netG.state_dict()
        netD_state_dict = netD.state_dict()
        newG_state_dict = {k: v for k, v in checkpointG["state_dict"].items() if k in netG_state_dict}
        newD_state_dict = {k: v for k, v in checkpointD["state_dict"].items() if k in netD_state_dict}
        # newG_state_dict = {k: v for k, v in checkpointG.items() if k in netG_state_dict}
        # newD_state_dict = {k: v for k, v in checkpointD.items() if k in netD_state_dict}
        # Overwrite the pretrained model weights to the current model
        netG_state_dict.update(newG_state_dict)
        netD_state_dict.update(newD_state_dict)
        netG.load_state_dict(netG_state_dict)
        netD.load_state_dict(netD_state_dict)
        print("Loaded pretrained model weights.")

    for epoch in range(config.teacher_start_epoch, config.epochs + 1):

        train_bar = tqdm(train_loader)
        running_results = {'batch_sizes': 0, 'loss': 0}

        student.train()
        for lr, hr in train_bar:
            lr = lr.to(device)
            hr = hr.to(device)
            batch_size = lr.size(0)
            running_results['batch_sizes'] += batch_size
            optimizer.zero_grad()
            student_fms, student_sr = student(lr)
            teacher_fms, teacher_sr = teacher(lr)

            aggregated_student_fms = []
            aggregated_teacher_fms = []

            if 'SA' in feature_distilation_type:
                aggregated_student_fms.append([spatial_similarity(fm) for fm in student_fms])
                aggregated_teacher_fms.append([spatial_similarity(fm) for fm in teacher_fms])
            if 'CA' in feature_distilation_type:
                aggregated_student_fms.append([channel_similarity(fm) for fm in student_fms])
                aggregated_teacher_fms.append([channel_similarity(fm) for fm in teacher_fms])
            if 'IA' in feature_distilation_type:
                aggregated_student_fms.append([batch_similarity(fm) for fm in student_fms])
                aggregated_teacher_fms.append([batch_similarity(fm) for fm in teacher_fms])
            if 'FSP' in feature_distilation_type:
                aggregated_student_fms.append(
                    [FSP(student_fms[i], student_fms[i + 1]) for i in range(len(student_fms) - 1)])
                aggregated_teacher_fms.append(
                    [FSP(teacher_fms[i], teacher_fms[i + 1]) for i in range(len(teacher_fms) - 1)])
            if 'AT' in feature_distilation_type:
                aggregated_student_fms.append([AT(fm) for fm in student_fms])
                aggregated_teacher_fms.append([AT(fm) for fm in teacher_fms])
            if 'fitnet' in feature_distilation_type:
                aggregated_student_fms.append([fm for fm in student_fms])
                aggregated_teacher_fms.append([fm for fm in teacher_fms])

            total_loss = criterion(student_sr, teacher_sr, hr, aggregated_student_fms, aggregated_teacher_fms)

            total_loss.backward()
            optimizer.step()

            # loss for current batch before optimization
            running_results['loss'] += total_loss.item() * batch_size

            train_bar.set_description(desc='[%d/%d] Loss: %.4f ' % (
                epoch, NUM_EPOCHS, running_results['loss'] / running_results['batch_sizes']))

        student.eval()
        out_path = 'training_results/SRF_' + str(UPSCALE_FACTOR) + '/'
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        with torch.no_grad():
            val_bar = tqdm(test_loader)
            valing_results = {'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'batch_sizes': 0}
            val_images = []
            for val_lr, val_hr_restore, val_hr in val_bar:
                batch_size = val_lr.size(0)
                valing_results['batch_sizes'] += batch_size
                lr = val_lr
                hr = val_hr
                if torch.cuda.is_available():
                    lr = lr.cuda()
                    hr = hr.cuda()
                _, sr = student(lr)

                batch_mse = ((sr - hr) ** 2).data.mean()
                valing_results['mse'] += batch_mse * batch_size
                batch_ssim = pytorch_ssim.ssim(sr, hr).item()
                valing_results['ssims'] += batch_ssim * batch_size
                valing_results['psnr'] = 10 * log10(
                    (hr.max() ** 2) / (valing_results['mse'] / valing_results['batch_sizes']))
                valing_results['ssim'] = valing_results['ssims'] / valing_results['batch_sizes']
                val_bar.set_description(
                    desc='[converting LR images to SR images] PSNR: %.4f dB SSIM: %.4f' % (
                        valing_results['psnr'], valing_results['ssim']))

            #     val_images.extend(
            #         [display_transform()(val_hr_restore.squeeze(0)), display_transform()(hr.data.cpu().squeeze(0)),
            #          display_transform()(sr.data.cpu().squeeze(0))])
            # val_images = torch.stack(val_images)
            # val_images = torch.chunk(val_images, val_images.size(0) // 15)
            # val_save_bar = tqdm(val_images, desc='[saving training results]')
            # index = 1
            # for image in val_save_bar:
            #     image = utils.make_grid(image, nrow=3, padding=5)
            #     utils.save_image(image, out_path + 'epoch_%d_index_%d.png' % (epoch, index), padding=5)
            #     index += 1

        # save model parameters
        if epoch % 5 == 0:
            torch.save({"epoch": epoch, "state_dict": student.state_dict()}, 'epochs/student_epoch_%d_%d.pth' % (UPSCALE_FACTOR, epoch))
        # save loss\scores\psnr\ssim
        results['loss'].append(running_results['loss'] / running_results['batch_sizes'])
        results['psnr'].append(valing_results['psnr'])
        results['ssim'].append(valing_results['ssim'])

        if epoch % 5 == 0 and epoch != 0:
            out_path = 'statistics/'
            data_frame = pd.DataFrame(
                data={'Loss': results['loss'], 'PSNR': results['psnr'], 'SSIM': results['ssim']},
                index=range(1, epoch + 1))
            data_frame.to_csv(out_path + 'srf_' + str(UPSCALE_FACTOR) + f'_train_results{config.teacher_start_epoch}.csv', index_label='Epoch')
