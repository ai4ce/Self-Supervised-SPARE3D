from __future__ import print_function, division

import torch
import argparse
import os
import numpy as np
import time
from model import Three2I_colorization
from Dataloader import *
from tensorboardX import SummaryWriter



parser = argparse.ArgumentParser()
parser.add_argument('--Training_dataroot', default="/home/wenyuhan/project/Train_dataset/Task_1_train_modify/",required=False, help='path to training dataset')
parser.add_argument('--Validating_dataroot', default="/home/wenyuhan/project/Train_dataset/Task_1_eval_modify/",required=False, help='path to validating dataset')
parser.add_argument('--model_root', default="/home/wenyuhan/project/Train_dataset/Task_1_eval_modify/",required=False, help='path to validating dataset')
parser.add_argument('--batchSize', type=int, default=8, help='input batch size')
parser.add_argument('--niter', type=int, default=50, help='number of epochs to train for')
parser.add_argument('--num_workers', type=int, default=1, help='number of workers')
parser.add_argument('--lr', type=float, default=0.00002, help='learning rate, default=0.00002')
parser.add_argument('--device', default='cuda', help='device')
parser.add_argument('--model_type', default='vgg16', help='|vgg16| |resnet50| |Bagnet33|')
parser.add_argument('--pretrained', action='store_true', default=False, help='If True, load pretrained dict')
parser.add_argument('--outf', default='/home/wenyuhan/final/3V2I_BC/', help='folder to output log')

opt = parser.parse_args()


device = opt.device
  
task_1_model = Three2I_colorization(opt.model_type, opt.pretrained).to(opt.device)
save_model = torch.load(os.path.join(opt.model_root, 'vgg16_Lr_1e-05.pth'))
model_dict = task_1_model.model_f.state_dict()
state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
t={}
for dic in model_dict.keys():
    if dic!='0.weight':
        new_dic='low_lv_feat_net.model.'+dic
        for k,v in save_model.items():
            if k==new_dic:
                t.update({dic:v})
model_dict.update(t)
task_1_model.model_f.load_state_dict(model_dict)
task_1_model.model_r.load_state_dict(model_dict)
task_1_model.model_t.load_state_dict(model_dict)
task_1_model.model_i.load_state_dict(model_dict)


def train_model():
    epoch_loss = 0
    epoch_acc =0 
    batch_loss = 0
    batch_acc =0
    batch_loss_list=[]
    path=opt.Training_dataroot
    train_data=ThreeV2I_BC_data(path)
    data_train = torch.utils.data.DataLoader(train_data,batch_size=opt.batchSize,pin_memory=True,num_workers=opt.num_workers, shuffle=True)
    task_1_model.train()
    for i, (Front_img,Right_img,Top_img,Ans_1,Ans_2,Ans_3,Ans_4,Label) in enumerate(data_train):
        optimizer.zero_grad()
        Front_img,Right_img,Top_img,Ans_1,Ans_2,Ans_3,Ans_4,Label=Front_img.to(device),Right_img.to(device),Top_img.to(device),Ans_1.to(device),Ans_2.to(device),Ans_3.to(device),Ans_4.to(device),Label.to(device)
        y1,y2,y3,y4 = task_1_model(Front_img.float(), Right_img.float(), Top_img.float(), Ans_1.float(), Ans_2.float(),
                         Ans_3.float(), Ans_4.float())
        Y = torch.cat((y1, y2, y3, y4), axis=1)
        loss = criterion(Y, Label.float())
        batch_loss += loss.item() * Front_img.shape[0]
        batch_loss_list.append(loss.item()*Front_img.shape[0]/len(train_data))
        loss.backward()
        optimizer.step()
        batch_acc += (Y.argmax(1) == Label.argmax(1)).sum().item()
    epoch_loss = batch_loss / len(train_data)
    epoch_acc =  batch_acc / len(train_data)
    
    return epoch_loss, epoch_acc, np.array(batch_loss_list)
        
        
                                        
def Eval():
    eval_loss = 0
    eval_acc = 0
    epoch_eval_loss=0
    epoch_eval_acc =0

    data_transforms=False
    path=opt.Validating_dataroot
    eval_data=ThreeV2I_BC_data(path)
    data_eval = torch.utils.data.DataLoader(eval_data,batch_size=opt.batchSize,pin_memory=True,num_workers=opt.num_workers, shuffle=True)
    with torch.no_grad():
        task_1_model.eval()
        for i, (Front_img,Right_img,Top_img,Ans_1,Ans_2,Ans_3,Ans_4,Label) in enumerate(data_eval):
            Front_img, Right_img, Top_img, Ans_1, Ans_2, Ans_3, Ans_4, Label = Front_img.to(device), Right_img.to(
                device), Top_img.to(device), Ans_1.to(device), Ans_2.to(device), Ans_3.to(device), Ans_4.to(
                device), Label.to(device)
            y1,y2,y3,y4= task_1_model(Front_img.float(), Right_img.float(), Top_img.float(), Ans_1.float(), Ans_2.float(), Ans_3.float(), Ans_4.float())
            Y = torch.cat((y1, y2, y3, y4), axis=1)
            loss = criterion(Y, Label.float())
            eval_loss += loss.item()* Front_img.shape[0]
            eval_acc += (Y.argmax(1) == Label.argmax(1)).sum().item()
        epoch_eval_loss = eval_loss / len(eval_data)
        epoch_eval_acc =eval_acc / len(eval_data)
    return epoch_eval_loss, epoch_eval_acc




N_EPOCHS = opt.niter
criterion = torch.nn.BCEWithLogitsLoss().to(opt.device)
optimizer = torch.optim.Adam(task_1_model.parameters(), lr=opt.lr)
#optimizer=torch.nn.DataParallel(optimizer)

batch_loss_history=[]



log_path=opt.outf
if os.path.exists(log_path)==False:
    os.makedirs(log_path)



file=open(log_path+"/"+opt.model_type+"_Lr_"+str(opt.lr)+".txt","w")

high_score=0
writer = SummaryWriter(opt.outf)

for epoch in range(N_EPOCHS):
    start_time = time.time()
    train_loss, train_acc,batch_list = train_model()
    valid_loss, valid_acc = Eval()
    batch_loss_history.append(batch_list)
    writer.add_scalar('train_loss', train_loss, global_step=epoch)
    writer.add_scalar('train_acc', train_acc, global_step=epoch)
    writer.add_scalar('test_loss', valid_loss, global_step=epoch)
    writer.add_scalar('test_acc', valid_acc, global_step=epoch)
    if high_score<valid_acc:
        torch.save(task_1_model.state_dict(), log_path + "/" + opt.model_type + "_Lr_" + str(opt.lr) + ".pth")
        high_score=valid_acc
        train_score=train_acc
    secs = int(time.time() - start_time)
    mins = secs / 60
    secs = secs % 60
    file.write('Epoch: %d' %(epoch + 1))
    file.write(" | time in %d minutes, %d seconds\n" %(mins, secs))
    file.write(f'\tLoss: {train_loss:.4f}(train)\t|\tAcc: {train_acc * 100:.1f}%(train)\n')
    file.write(f'\tLoss: {valid_loss:.4f}(valid)\t|\tAcc: {valid_acc * 100:.1f}%(valid)\n')
    file.write("\n")
    file.flush()

writer.close()
file.close()
file = open(log_path + "/" + "lowest_loss.txt", "w")
file.write(f'\tAcc: {train_score * 100:.1f}%(train)\tAcc: {high_score * 100:.1f}%(valid)\n')
file.close()

batch_loss_history=np.array(batch_loss_history)
batch_loss_history=np.concatenate(batch_loss_history,axis=0)
batch_loss_history=batch_loss_history.reshape(len(batch_loss_history))
np.save(log_path+"/"+opt.model_type+"_Lr_"+str(opt.lr)+".npy",batch_loss_history)


