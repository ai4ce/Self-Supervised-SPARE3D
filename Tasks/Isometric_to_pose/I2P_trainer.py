from __future__ import print_function, division

import torch
import argparse
import os
import numpy as np
import time

from Dataloader import I2P_data
from model import *
#from Dataloader import *
from tensorboardX import SummaryWriter



parser = argparse.ArgumentParser()
parser.add_argument('--Training_dataroot', default="/home/wenyuhan/project/Train_dataset/Task_4_train",required=False, help='path to training dataset')
parser.add_argument('--Validating_dataroot', default="/home/wenyuhan/project/Train_dataset/Task_4_eval",required=False, help='path to validating dataset')
parser.add_argument('--batchSize', type=int, default=70, help='input batch size')
parser.add_argument('--niter', type=int, default=50, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.00005, help='learning rate, default=0.00005')
parser.add_argument('--device', default='cuda:0', help='device')
parser.add_argument('--model_type', default='vgg16', help='|vgg16| |resnet50| |Bagnet33|')
parser.add_argument('--pretrained', action='store_true', default=False, help='If True, load pretrained dict')
parser.add_argument('--outf', default='/home/wenyuhan/final/I2P/', help='folder to output log')

opt = parser.parse_args()


device = opt.device

task_4_model=I2P(opt.model_type,opt.pretrained).to(opt.device)
#task_4_model=nn.DataParallel(task_4_model).to(opt.device)


def train_model():
    batch_loss = 0
    batch_acc =0
    batch_loss_list=[]
    path=opt.Training_dataroot
    train_data = I2P_data(path)
    data_train = torch.utils.data.DataLoader(train_data,batch_size=opt.batchSize,pin_memory=True,num_workers=4, shuffle=True)
    task_4_model.train()
    for i, (Dic,Inputf,Inputr,Inputt,Inputa,Label) in enumerate(data_train):
        optimizer.zero_grad()
        Inputf,Inputr,Inputt,Inputa,Label=Inputf.to(device),Inputr.to(device),Inputt.to(device),Inputa.to(device),Label.to(device)
        y = task_4_model(Inputf.float(),Inputr.float(),Inputt.float(),Inputa.float())
        Label=Label.reshape(Inputf.shape[0])
        loss = criterion(y, Label)
        batch_loss += loss.item() * Inputf.shape[0]
        batch_loss_list.append(loss.item()*Inputf.shape[0]/len(train_data))
        loss.backward()
        #optimizer.module.step()
        optimizer.step()
        batch_acc += (y.argmax(1) == Label).sum().item()
    epoch_loss = batch_loss / len(train_data)
    epoch_acc =  batch_acc /len(train_data)
    return epoch_loss, epoch_acc, np.array(batch_loss_list)


def Eval():
    eval_loss = 0
    eval_acc = 0
    path=opt.Validating_dataroot
    eval_data = I2P_data(path)
    data_eval = torch.utils.data.DataLoader(eval_data,batch_size=opt.batchSize,pin_memory=True,num_workers=4, shuffle=True)
    failed_shape=[]
    with torch.no_grad():
        task_4_model.eval()
        for i, (Dic,Inputf,Inputr,Inputt,Inputa,Label) in enumerate(data_eval):
            Inputf,Inputr,Inputt,Inputa,Label=Inputf.to(device),Inputr.to(device),Inputt.to(device),Inputa.to(device),Label.to(device)
            y = task_4_model(Inputf.float(),Inputr.float(),Inputt.float(),Inputa.float())
            Label=Label.reshape(Inputf.shape[0])
            loss = criterion(y, Label)
            Dic=np.array(Dic)
            Dic=Dic[(y.argmax(1)!=Label).cpu()]
            for dic in Dic:
                failed_shape.append(dic)
            eval_loss += loss.item()* Inputf.shape[0]
            eval_acc += (y.argmax(1) == Label).sum().item()
        epoch_eval_loss = eval_loss / len(eval_data)
        epoch_eval_acc =eval_acc / len(eval_data)
    return failed_shape,epoch_eval_loss, epoch_eval_acc




N_EPOCHS = opt.niter
criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(task_4_model.parameters(), lr=opt.lr)
#optimizer=nn.DataParallel(optimizer)

batch_loss_history=[]



log_path=opt.outf
if os.path.exists(log_path)==False:
    os.makedirs(log_path)

writer =SummaryWriter(opt.outf)

file=open(log_path+"/"+opt.model_type+"_Lr_"+str(opt.lr)+".txt","w")
bad_shape=[]

high_valid_acc = 0
high_train_acc=0

for epoch in range(N_EPOCHS):
    start_time = time.time()
    train_loss, train_acc,batch_list = train_model()
    failed_shape,valid_loss, valid_acc = Eval()
    if high_valid_acc<valid_acc:
        torch.save(task_4_model.state_dict(), log_path + "/" + opt.model_type + "_Lr_" + str(opt.lr) + ".pth")
        bad_shape=failed_shape
        high_valid_acc=valid_acc
        high_train_acc=train_acc
    batch_loss_history.append(batch_list)
    secs = int(time.time() - start_time)
    mins = secs / 60
    secs = secs % 60
    writer.add_scalar('train_loss', train_loss, global_step=epoch)
    writer.add_scalar('train_acc', train_acc, global_step=epoch)
    writer.add_scalar('test_loss', valid_loss, global_step=epoch)
    writer.add_scalar('test_acc', valid_acc, global_step=epoch)
    file.write('Epoch: %d' %(epoch + 1))
    file.write(" | time in %d minutes, %d seconds\n" %(mins, secs))
    file.write(f'\tLoss: {train_loss:.4f}(train)\t|\tAcc: {train_acc * 100:.1f}%(train)\n')
    file.write(f'\tLoss: {valid_loss:.4f}(valid)\t|\tAcc: {valid_acc * 100:.1f}%(valid)\n')
    file.write("\n")
    file.flush()

file.close()

file=open(log_path+"/"+"high.txt","w")
file.write(f'\tAcc: {high_train_acc * 100:.1f}%(train)\tAcc: {high_valid_acc * 100:.1f}%(valid)\n')
file.close()

writer.close()

batch_loss_history=np.array(batch_loss_history)
batch_loss_history=np.concatenate(batch_loss_history,axis=0)
    
batch_loss_history=batch_loss_history.reshape(len(batch_loss_history))
np.save(log_path+"/"+opt.model_type+"_Lr_"+str(opt.lr)+".npy",batch_loss_history)


