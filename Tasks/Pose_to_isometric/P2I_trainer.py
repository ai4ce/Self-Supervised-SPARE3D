from __future__ import print_function, division
import torch.nn as nn
import torch
import argparse
import os
import numpy as np
from tensorboardX import SummaryWriter
import time
from model import P2I
from Dataloader import P2I_data

parser = argparse.ArgumentParser()
parser.add_argument('--Training_dataroot',
                    default="C:/Users/ay162\Desktop/research/New_data\P2I\P2I_data\Distance\Test_Group0/train",
                    required=False,
                    help='path to training dataset')
parser.add_argument('--Validating_dataroot',
                    default="C:/Users/ay162\Desktop/research/New_data\P2I\P2I_data\Distance\Test_Group0/test",
                    required=False,
                    help='path to validating dataset')
parser.add_argument('--batchSize', type=int, default=30, help='input batch size')
parser.add_argument('--niter', type=int, default=50, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.00005, help='learning rate, default=0.00002')
parser.add_argument('--width', type=int, default=64, help='model width')
parser.add_argument('--device', default='cuda:0', help='device')
parser.add_argument('--model_type', default='vgg16', help='|vgg16| |resnet50| |Bagnet33|')
parser.add_argument('--pretrained', action='store_true', default=False, help='If True, load pretrained dict')
parser.add_argument('--outf', default='C:/Users/ay162\Desktop/research/New_data\P2I\P2I_data\Distance\Test_Group0/',
                    help='folder to output log')
opt = parser.parse_args()
device = opt.device
task_5_model = P2I(opt.model_type,opt.pretrained).to(opt.device)


def train_model():
    batch_loss = 0
    batch_acc = 0
    batch_loss_list = []
    path = opt.Training_dataroot
    train_data = P2I_data(path)
    data_train = torch.utils.data.DataLoader(train_data, batch_size=opt.batchSize,pin_memory=True,num_workers=1, shuffle=True)
    task_5_model.train()
    for i, (Front_img, Right_img, Top_img, Ans_1, Ans_2, Ans_3, Ans_4, Label, View_vector) in enumerate(data_train):
        optimizer.zero_grad()
        Front_img, Right_img, Top_img, Ans_1, Ans_2, Ans_3, Ans_4, Label, View_vector = Front_img.to(
            device), Right_img.to(device), Top_img.to(device), Ans_1.to(device), Ans_2.to(device), Ans_3.to(
            device), Ans_4.to(device), Label.to(device), View_vector.to(device)
        y_1, y_2, y_3, y_4 = task_5_model(Front_img.float(), Right_img.float(), Top_img.float(), Ans_1.float(),
                                          Ans_2.float(), Ans_3.float(), Ans_4.float())
        y_1 = y_1[View_vector].reshape(Front_img.shape[0], 1)
        y_2 = y_2[View_vector].reshape(Front_img.shape[0], 1)
        y_3 = y_3[View_vector].reshape(Front_img.shape[0], 1)
        y_4 = y_4[View_vector].reshape(Front_img.shape[0], 1)
        Y = torch.cat((y_1, y_2, y_3, y_4), axis=1)
        loss = criterion(Y, Label.float())
        batch_loss += loss.item() * Front_img.shape[0]
        batch_loss_list.append(loss.item() * Front_img.shape[0] / len(train_data))
        loss.backward()
        optimizer.step()
        batch_acc += (Y.argmax(1) == Label.argmax(1)).sum().item()
    epoch_loss = batch_loss / len(train_data)
    epoch_acc = batch_acc / len(train_data)
    return epoch_loss, epoch_acc, np.array(batch_loss_list)


def Eval():
    eval_loss = 0
    eval_acc = 0
    path = opt.Validating_dataroot
    eval_data = P2I_data(path)
    data_eval = torch.utils.data.DataLoader(eval_data, batch_size=opt.batchSize,pin_memory=True,num_workers=1, shuffle=True)

    with torch.no_grad():
        task_5_model.eval()
        for i, (Front_img, Right_img, Top_img, Ans_1, Ans_2, Ans_3, Ans_4, Label, View_vector) in enumerate(data_eval):
            Front_img, Right_img, Top_img, Ans_1, Ans_2, Ans_3, Ans_4, Label, View_vector = Front_img.to(
                device), Right_img.to(device), Top_img.to(device), Ans_1.to(device), Ans_2.to(device), Ans_3.to(
                device), Ans_4.to(device), Label.to(device), View_vector.to(device)
            y_1, y_2, y_3, y_4 = task_5_model(Front_img.float(), Right_img.float(), Top_img.float(), Ans_1.float(),
                                              Ans_2.float(), Ans_3.float(), Ans_4.float())
            y_1 = y_1[View_vector].reshape(Front_img.shape[0], 1)
            y_2 = y_2[View_vector].reshape(Front_img.shape[0], 1)
            y_3 = y_3[View_vector].reshape(Front_img.shape[0], 1)
            y_4 = y_4[View_vector].reshape(Front_img.shape[0], 1)
            Y = torch.cat((y_1, y_2, y_3, y_4), axis=1)
            loss = criterion(Y, Label.float())
            eval_loss += loss.item() * Front_img.shape[0]
            eval_acc += (Y.argmax(1) == Label.argmax(1)).sum().item()
        epoch_eval_loss = eval_loss / len(eval_data)
        epoch_eval_acc = eval_acc / len(eval_data)
    return epoch_eval_loss, epoch_eval_acc


N_EPOCHS = opt.niter
criterion = torch.nn.BCEWithLogitsLoss().to(device)
optimizer = torch.optim.Adam(task_5_model.parameters(), lr=opt.lr)
batch_loss_history = []
log_path = opt.outf
writer = SummaryWriter(opt.outf)

if os.path.exists(log_path) == False:
    os.makedirs(log_path)

file = open(log_path + "/" + opt.model_type + "_BC.txt", "w")
high_score = 0

for epoch in range(N_EPOCHS):
    start_time = time.time()
    train_loss, train_acc, batch_list = train_model()
    valid_loss, valid_acc = Eval()
    if high_score < valid_acc:
        high_score = valid_acc
        train_high=train_acc
        torch.save(task_5_model.state_dict(), log_path + "/" + opt.model_type + "_Lr_" + str(opt.lr) + ".pth")
    batch_loss_history.append(batch_list)
    secs = int(time.time() - start_time)
    mins = secs / 60
    secs = secs % 60
    writer.add_scalar('train_loss', train_loss, global_step=epoch)
    writer.add_scalar('train_acc', train_acc, global_step=epoch)
    writer.add_scalar('test_loss', valid_loss, global_step=epoch)
    writer.add_scalar('test_acc', valid_acc, global_step=epoch)
    file.write('Epoch: %d' % (epoch + 1))
    file.write(" | time in %d minutes, %d seconds\n" % (mins, secs))
    file.write(f'\tLoss: {train_loss:.4f}(train)\t|\tAcc: {train_acc * 100:.1f}%(train)\n')
    file.write(f'\tLoss: {valid_loss:.4f}(valid)\t|\tAcc: {valid_acc * 100:.1f}%(valid)\n')
    file.write("\n")
    file.flush()
file.close()
writer.close()

file=open(log_path+"/"+"high.txt","w")
file.write(f'\tAcc: {train_high * 100:.1f}%(train)\tAcc: {high_score * 100:.1f}%(valid)\n')
file.close()

batch_loss_history = np.array(batch_loss_history)
batch_loss_history = np.concatenate(batch_loss_history, axis=0)
batch_loss_history = batch_loss_history.reshape(len(batch_loss_history))
# np.save(log_path + "/" + opt.model_type + "_Lr_" + str(opt.lr) + ".npy", batch_loss_history)
