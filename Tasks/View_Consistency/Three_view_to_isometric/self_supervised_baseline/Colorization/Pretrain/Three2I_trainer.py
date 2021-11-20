from __future__ import print_function, division

import torch
import torch.nn as nn
import argparse
import os
import numpy as np
import time
from model import ColorNet
from Dataloader import ThreeV2I_data
from tensorboardX import SummaryWriter


parser = argparse.ArgumentParser()
parser.add_argument('--Training_dataroot', default="C:/Users/ay162\Desktop\slack\color/train",
                    required=False, help='path to training dataset')
parser.add_argument('--Validating_dataroot', default="C:/Users/ay162\Desktop\slack\color/valid",
                    required=False, help='path to validating dataset')
parser.add_argument('--batchSize', type=int, default=3, help='input batch size')
parser.add_argument('--niter', type=int, default=50, help='number of epochs to train for')
parser.add_argument('--num_workers', type=int, default=3, help='input batch size')
parser.add_argument('--lr', type=float, default=0.00002, help='learning rate, default=0.00002')
parser.add_argument('--pretrained', action='store_true', default=False, help='If True, load pretrained dict')
parser.add_argument('--device', default='cuda:0', help='device')
parser.add_argument('--model_type', default='vgg16', help='|vgg16| |resnet50| |Bagnet33|')
parser.add_argument('--outf', default='C:/Users/ay162\Desktop\slack\color/', help='folder to output log')


opt = parser.parse_args()
device = opt.device

task_1_model = ColorNet(opt.pretrained).to(device)


def train_model():
    batch_loss = 0
    path = opt.Training_dataroot
    train_data = ThreeV2I_data(path)
    data_train = torch.utils.data.DataLoader(train_data, batch_size=opt.batchSize,pin_memory=True, num_workers=opt.num_workers,
                                             shuffle=True)
    task_1_model.train()
    for i, (gray_img,original_img) in enumerate(data_train):
        optimizer.zero_grad()
        gray_img,original_img= gray_img.to(device),original_img.to(device)
        pre_img= task_1_model(gray_img.float(), gray_img.float())
        loss =  torch.pow((original_img - pre_img), 2).sum() / torch.from_numpy(np.array(list(pre_img.size()))).prod()
        batch_loss += loss.item() * gray_img.shape[0]
        loss.backward()
        optimizer.step()
    epoch_loss = batch_loss / len(train_data)

    return epoch_loss


def Eval():
    eval_loss = 0
    path = opt.Validating_dataroot
    eval_data = ThreeV2I_data(path)
    data_eval = torch.utils.data.DataLoader(eval_data, batch_size=opt.batchSize,pin_memory=True, num_workers=opt.num_workers,
                                            shuffle=True)
    with torch.no_grad():
        task_1_model.eval()
        for i, (gray_img,original_img)in enumerate(data_eval):
            gray_img, original_img = gray_img.to(device), original_img.to(device)
            pre_img = task_1_model(gray_img.float(), gray_img.float())
            loss = torch.pow((original_img - pre_img), 2).sum() / torch.from_numpy(
                np.array(list(pre_img.size()))).prod()
            eval_loss += loss.item() * gray_img.shape[0]
        epoch_eval_loss = eval_loss / len(eval_data)
    return epoch_eval_loss


N_EPOCHS = opt.niter
criterion = torch.nn.BCEWithLogitsLoss().to(opt.device)
optimizer = torch.optim.Adam(task_1_model.parameters(), lr=opt.lr)
#optimizer=nn.DataParallel(optimizer)
batch_loss_history = []

log_path = opt.outf
if os.path.exists(log_path) == False:
    os.makedirs(log_path)

file = open(log_path + "/" + opt.model_type + "_Lr_" + str(opt.lr) + ".txt", "w")

lower_valid_loss = np.inf
writer = SummaryWriter(opt.outf)


for epoch in range(N_EPOCHS):
    start_time = time.time()
    train_loss = train_model()
    valid_loss = Eval()
    writer.add_scalar('train_loss', train_loss, global_step=epoch)
    writer.add_scalar('test_loss', valid_loss, global_step=epoch)
    if lower_valid_loss > valid_loss:
        torch.save(task_1_model.state_dict(), log_path + "/" + opt.model_type + "_Lr_" + str(opt.lr) + ".pth")
        lower_valid_loss = valid_loss
        lower_train_loss = train_loss

    secs = int(time.time() - start_time)
    mins = secs / 60
    secs = secs % 60
    file.write('Epoch: %d' % (epoch + 1))
    file.write(" | time in %d minutes, %d seconds\n" % (mins, secs))
    file.write(f'\tLoss: {train_loss:.4f}(train)\t|\tLoss: {valid_loss:.4f}(valid)\n')
    file.write("\n")
    file.flush()

file.close()
writer.close()

file = open(log_path + "/" + "lowest_loss.txt", "w")
file.write(f'\tAcc: {lower_train_loss :.3f}%(train)\tAcc: {lower_valid_loss :.3f}%(valid)\n')
file.close()
torch.save(task_1_model.state_dict(), log_path + "/finial.pth")
