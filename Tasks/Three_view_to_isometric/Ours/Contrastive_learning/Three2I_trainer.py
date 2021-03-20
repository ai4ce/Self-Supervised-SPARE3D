from __future__ import print_function, division

import torch
import torch.nn as nn
import argparse
import os
import numpy as np
import time
from model import Three2I_self
from Dataloader import ThreeV2I_data
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument('--Training_dataroot', default="C:/Users/ay162\Desktop\slack\data/train",
                    required=False, help='path to training dataset')
parser.add_argument('--Validating_dataroot', default="C:/Users/ay162\Desktop\slack\data/valid",
                    required=False, help='path to validating dataset')
parser.add_argument('--Testing_dataroot', default="C:/Users/ay162\Desktop\slack\data/test",
                    required=False, help='path to validating dataset')
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--niter', type=int, default=1, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.00002, help='learning rate, default=0.00002')
parser.add_argument('--pretrained', action='store_true', default=False, help='If True, load pretrained dict')
parser.add_argument('--device', default='cuda:0', help='device')
parser.add_argument('--model_type', default='vgg16', help='|vgg16| |resnet50| |Bagnet33|')
parser.add_argument('--num_workers', type=int, default=1, help='number of workers')
parser.add_argument('--outf', default='C:/Users/ay162\Desktop\slack\data', help='folder to output log')

opt = parser.parse_args()

device = opt.device

task_1_model = Three2I_self(opt.model_type,opt.pretrained).to(device)
#task_1_model=nn.DataParallel(task_1_model).to(device)

def train_model():
    batch_loss = 0
    path = opt.Training_dataroot
    train_data = ThreeV2I_data(path)
    data_train = torch.utils.data.DataLoader(train_data, batch_size=opt.batchSize, pin_memory=True, num_workers=opt.num_workers,
                                             shuffle=True)
    task_1_model.train()
    for i, (
            Front_imgl, Right_imgl, Top_imgl, Isometricl, Front_imgr, Right_imgr, Top_imgr, Isometricr,
            Label) in enumerate(
        data_train):
        optimizer.zero_grad()
        Front_imgl, Right_imgl, Top_imgl, Isometricl, Front_imgr, Right_imgr, Top_imgr, Isometricr, Label = Front_imgl.to(
            device), Right_imgl.to(device), Top_imgl.to(device), Isometricl.to(device), Front_imgr.to(
            device), Right_imgr.to(device), Top_imgr.to(device), Isometricr.to(device), Label.to(device)
        y1, y2, y3, y4 = task_1_model(Front_imgl.float(), Right_imgl.float(), Top_imgl.float(), Isometricl.float(),
                                      Front_imgr.float(), Right_imgr.float(), Top_imgr.float(),
                                      Isometricr.float(),False)
        Y = torch.cat((y1, y2, y3, y4), axis=1)
        loss = criterion(Y, Label.float())
        batch_loss += loss.item() * Front_imgl.shape[0]
        loss.backward()
        optimizer.step()
    epoch_loss = batch_loss / len(train_data)

    return epoch_loss


def Eval():
    eval_loss = 0
    path = opt.Validating_dataroot
    eval_data = ThreeV2I_data(path)
    data_eval = torch.utils.data.DataLoader(eval_data, batch_size=opt.batchSize, pin_memory=True, num_workers=opt.num_workers,
                                            shuffle=True)
    with torch.no_grad():
        task_1_model.eval()
        for i, (
                Front_imgl, Right_imgl, Top_imgl, Isometricl, Front_imgr, Right_imgr, Top_imgr, Isometricr,
                Label) in enumerate(
            data_eval):
            Front_imgl, Right_imgl, Top_imgl, Isometricl, Front_imgr, Right_imgr, Top_imgr, Isometricr, Label = Front_imgl.to(
                device), Right_imgl.to(device), Top_imgl.to(device), Isometricl.to(device), Front_imgr.to(
                device), Right_imgr.to(device), Top_imgr.to(device), Isometricr.to(device), Label.to(device)
            y1, y2, y3, y4 = task_1_model(Front_imgl.float(), Right_imgl.float(), Top_imgl.float(), Isometricl.float(),
                                          Front_imgr.float(), Right_imgr.float(), Top_imgr.float(),
                                          Isometricr.float(),False)
            Y = torch.cat((y1, y2, y3, y4), axis=1)
            loss = criterion(Y, Label.float())
            eval_loss += loss.item() * Front_imgl.shape[0]
        epoch_eval_loss = eval_loss / len(eval_data)
    return epoch_eval_loss

def Test():
    eval_loss = 0
    eval_acc = 0

    path = opt.Testing_dataroot
    test_data = Three2I_data(path)
    data_test = torch.utils.data.DataLoader(test_data, batch_size=1, pin_memory=True, num_workers=1,
                                            shuffle=True)

    with torch.no_grad():
        task_1_model.eval()
        for i, (Front_img, Right_img, Top_img, Ans_1, Ans_2, Ans_3, Ans_4, Label) in enumerate(data_test):
            Front_img, Right_img, Top_img, Ans_1, Ans_2, Ans_3, Ans_4, Label = Front_img.to(device), Right_img.to(
                device), Top_img.to(device), Ans_1.to(device), Ans_2.to(device), Ans_3.to(device), Ans_4.to(
                device), Label.to(device)
            y1, y2, y3, y4 = task_1_model(Front_img.float(), Right_img.float(), Top_img.float(), Ans_1.float(),
                                          Ans_2.float(), Ans_3.float(), Ans_4.float(),torch.tensor([0.]),True)
            Y = torch.cat((y1, y2, y3, y4), axis=1)
            loss = criterion(Y, Label.float())
            eval_loss += loss.item() * Front_img.shape[0]
            eval_acc += (Y.argmax(1) == Label.argmax(1)).sum().item()
        epoch_eval_loss = eval_loss / len(test_data)
        epoch_eval_acc = eval_acc / len(test_data)
    return epoch_eval_loss, epoch_eval_acc

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
    testing_loss,testing_acc=Test()
    writer.add_scalar('train_loss', train_loss, global_step=epoch)
    writer.add_scalar('test_loss', valid_loss, global_step=epoch)
    writer.add_scalar('test_acc', testing_acc, global_step=epoch)
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
    file.write(f'\tLoss: {testing_loss:.4f}(test)\t|\tAcc: {testing_acc * 100:.1f}%(test)\n')
    file.write("\n")
    file.flush()

file.close()
writer.close()

file = open(log_path + "/" + "lowest_loss.txt", "w")
file.write(f'\tAcc: {lower_train_loss :.3f}%(train)\tAcc: {lower_valid_loss :.3f}%(valid)\n')
file.close()
torch.save(task_1_model.state_dict(), log_path + "/finial.pth")