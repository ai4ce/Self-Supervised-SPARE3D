from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
import numpy as np
import time
from model import Three2I_SimCLR
from Dataloader import Three2I_SimCLR_data
from tensorboardX import SummaryWriter
from nt_xnet import NTXentLoss


parser = argparse.ArgumentParser()
parser.add_argument('--Training_dataroot', default="D:\spare3d_plus\Final_data\I2P_data/test/train", required=False,
                    help='path to training dataset')
parser.add_argument('--Validating_dataroot', default="D:\spare3d_plus\Final_data\I2P_data/test/valid", required=False,
                    help='path to validating dataset')
parser.add_argument('--batchSize', type=int, default=5, help='input batch size')
parser.add_argument('--niter', type=int, default=3, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.00002, help='learning rate, default=0.00002')
parser.add_argument('--device', default='cuda', help='device')
parser.add_argument('--model_type', default='vgg16', help='|vgg16| |resnet50| |Bagnet33|')
parser.add_argument('--pretrained', action='store_true', default=False, help='If True, load pretrained dict')
parser.add_argument('--outf', default='D:\spare3d_plus\Final_data\I2P_data/test', help='folder to output log')


opt = parser.parse_args()
device = opt.device

task_1_model = Three2I_SimCLR(opt.model_type,opt.pretrained)
task_1_model =nn.DataParallel(task_1_model).to(device)


def train_model():
    epoch_loss = 0
    epoch_acc = 0
    batch_loss = 0
    batch_loss_list = []
    path = opt.Training_dataroot
    train_data = Three2I_SimCLR_data(path)
    data_train = torch.utils.data.DataLoader(train_data, batch_size=opt.batchSize, pin_memory=True, num_workers=10,
                                             shuffle=True)
    task_1_model.train()
    for i, (Front_img, Right_img, Top_img, Ans) in enumerate(data_train):
        optimizer.zero_grad()
        batchsize = Front_img.shape[0]
        nt_xent_criterion = NTXentLoss(device, batchsize, 0.5, True)
        Front_img, Right_img, Top_img, Ans = Front_img.to(device), Right_img.to(device), Top_img.to(device), Ans.to(
            device)
        zis, zjs = task_1_model(Front_img.float(), Right_img.float(), Top_img.float(), Ans.float())
        zis = F.normalize(zis, dim=1)
        zjs = F.normalize(zjs, dim=1)
        loss = nt_xent_criterion(zis, zjs)
        batch_loss += loss.item()
        batch_loss_list.append(loss.item() * Front_img.shape[0] / len(train_data))
        loss.backward()
        optimizer.module.step()

    epoch_loss = batch_loss / len(train_data)

    return epoch_loss, np.array(batch_loss_list)


def Eval():
    eval_loss = 0
    epoch_eval_loss = 0
    epoch_eval_acc = 0

    data_transforms = False
    path = opt.Validating_dataroot
    eval_data = Three2I_SimCLR_data(path)
    data_eval = torch.utils.data.DataLoader(eval_data, batch_size=opt.batchSize, pin_memory=True, num_workers=10,
                                            shuffle=True)
    failed_shape = []
    with torch.no_grad():
        task_1_model.eval()
        for i, (Front_img, Right_img, Top_img, Ans) in enumerate(data_eval):
            batchsize = Front_img.shape[0]
            nt_xent_criterion = NTXentLoss(device, batchsize, 0.5, True)
            Front_img, Right_img, Top_img, Ans = Front_img.to(device), Right_img.to(device), Top_img.to(device), Ans.to(
                device)
            zis, zjs = task_1_model(Front_img.float(), Right_img.float(), Top_img.float(), Ans.float())
            zis = F.normalize(zis, dim=1)
            zjs = F.normalize(zjs, dim=1)
            loss = nt_xent_criterion(zis, zjs)
            eval_loss += loss.item()
        epoch_eval_loss = eval_loss / len(eval_data)
    return epoch_eval_loss


N_EPOCHS = opt.niter
optimizer = torch.optim.Adam(task_1_model.parameters(), lr=opt.lr, weight_decay=0.000001)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.batchSize, eta_min=0,
                                                       last_epoch=-1)
optimizer = nn.DataParallel(optimizer)

batch_loss_history = []

log_path = opt.outf
if os.path.exists(log_path) == False:
    os.makedirs(log_path)

writer = SummaryWriter(opt.outf)

file = open(log_path + "/" + opt.model_type + "_Lr_" + str(opt.lr) + ".txt", "w")

bad_shape = []

high_loss = np.inf


for epoch in range(N_EPOCHS):
    start_time = time.time()
    train_loss, batch_list = train_model()
    valid_loss = Eval()
    if epoch >= 20:
        scheduler.step()
    if high_loss > valid_loss:
        torch.save(task_1_model.module.state_dict(), log_path + "/" + opt.model_type + "_Lr_" + str(opt.lr) + ".pth")
        high_loss = valid_loss
    batch_loss_history.append(batch_list)
    secs = int(time.time() - start_time)
    mins = secs / 60
    secs = secs % 60
    writer.add_scalar('train_loss', train_loss, global_step=epoch)
    writer.add_scalar('test_loss', valid_loss, global_step=epoch)
    file.write('Epoch: %d' % (epoch + 1))
    file.write(" | time in %d minutes, %d seconds\n" % (mins, secs))
    file.write(f'\tLoss: {train_loss:.4f}(train)\n')
    file.write(f'\tLoss: {valid_loss:.4f}(valid)\n')
    file.write("\n")
    file.flush()

file.close()
writer.close()

batch_loss_history = np.array(batch_loss_history)
batch_loss_history = np.concatenate(batch_loss_history, axis=0)

batch_loss_history = batch_loss_history.reshape(len(batch_loss_history))
np.save(log_path + "/" + opt.model_type + "_Lr_" + str(opt.lr) + ".npy", batch_loss_history)


