from __future__ import print_function, division
import torch
import argparse
import os
from tensorboardX import SummaryWriter
import time
from model import SR
from Dataloader import I2P_data
import json
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument('--Dataroot', default=None, required=True, help='path to dataset')
parser.add_argument('--Dataset', default=".", required=True,
                    help='path to validating dataset')
parser.add_argument('--pretrained_model', default=None, required=False,
                    help='path to pretrained model')
parser.add_argument('--num_train', type=int, default=5000, help='number of training dataset')
parser.add_argument('--batchSize', type=int, default=30, help='input batch size')
parser.add_argument('--niter', type=int, default=50, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.00005, help='learning rate, default=0.00002')
parser.add_argument('--device', default='cuda:0', help='device')
parser.add_argument('--pretrained', action='store_true', default=False, help='If True, load pretrained dict')
parser.add_argument('--outf', default='.',
                    help='folder to output log')
opt = parser.parse_args()
device = opt.device

model=SR(opt.pretrained).to(opt.device)
if opt.pretrained_model:
    save_model = torch.load(opt.pretrained_model)
    model_dict = model.state_dict()
    state_dict = {k: v for k, v in save_model.items() if (k!='classifier.3.weight') and (k!='classifier.3.bias')}
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)
dataroot = opt.Dataroot
with open(opt.Dataset, 'r') as f:
    database = json.load(f)
dataset = database.get('I2P')
dataset_train = dataset.get('train')
dataset_valid_I2P = dataset.get('valid')
dataset_train_I2P = {}
for i in list(dataset_train.keys())[:opt.num_train]:
    dataset_train_I2P.update({i: dataset_train[i]})

def Train():
    train_loss = 0
    train_acc = 0
    train_data = I2P_data(os.path.join(dataroot,'train'),dataset_train_I2P)
    data_train = torch.utils.data.DataLoader(train_data,batch_size=opt.batchSize,pin_memory=True,num_workers=8, shuffle=True)
    model.train()
    for i, (Inputf,Inputr,Inputt,Inputa,Label) in enumerate(data_train):
        Inputf,Inputr,Inputt,Inputa,Label=Inputf.to(device),Inputr.to(device),Inputt.to(device),Inputa.to(device),Label.to(device)
        y = F.normalize(model(Inputf.float(),Inputr.float(),Inputt.float(),Inputa.float()),dim=1)
        Label=Label.reshape(Inputf.shape[0])
        loss = criterion_I2P(y, Label)
        train_loss += loss.item()* Inputf.shape[0]
        train_acc += (y.argmax(1) == Label).sum().item()
        loss.backward()
        optimizer.step()
    epoch_train_loss = train_loss / len(train_data)
    epoch_train_acc =train_acc / len(train_data)
    return epoch_train_loss, epoch_train_acc

def Eval():
    eval_loss = 0
    eval_acc = 0
    eval_data = I2P_data(os.path.join(dataroot,'valid'),dataset_valid_I2P)
    data_eval = torch.utils.data.DataLoader(eval_data,batch_size=opt.batchSize,pin_memory=True,num_workers=8, shuffle=True)
    with torch.no_grad():
        model.eval()
        for i, (Inputf,Inputr,Inputt,Inputa,Label) in enumerate(data_eval):
            Inputf,Inputr,Inputt,Inputa,Label=Inputf.to(device),Inputr.to(device),Inputt.to(device),Inputa.to(device),Label.to(device)
            y = F.normalize(model(Inputf.float(),Inputr.float(),Inputt.float(),Inputa.float()),dim=1)
            Label=Label.reshape(Inputf.shape[0])
            loss = criterion_I2P(y, Label)
            eval_loss += loss.item()* Inputf.shape[0]
            eval_acc += (y.argmax(1) == Label).sum().item()
        epoch_eval_loss = eval_loss / len(eval_data)
        epoch_eval_acc =eval_acc / len(eval_data)
    return epoch_eval_loss, epoch_eval_acc

N_EPOCHS = opt.niter
criterion = torch.nn.BCEWithLogitsLoss().to(device)
criterion_I2P = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
log_path = opt.outf
writer = SummaryWriter(opt.outf)

if os.path.exists(log_path) == False:
    os.makedirs(log_path)

file = open(log_path + "/I2P_" + str(opt.lr) + ".txt", "w")
I2P_high_score = 0

for epoch in range(N_EPOCHS):
    start_time = time.time()
    I2P_train_loss, I2P_train_acc= Train()
    I2P_valid_loss, I2P_valid_acc = Eval()

    if I2P_high_score < I2P_valid_acc:
        I2P_high_score = I2P_valid_acc
        torch.save(model.state_dict(), log_path + "/I2P_Lr_" + str(opt.lr) + ".pth")
    secs = int(time.time() - start_time)
    mins = secs / 60
    secs = secs % 60
    writer.add_scalar('I2P_loss', I2P_train_loss, global_step=epoch)
    writer.add_scalar('I2P_acc', I2P_valid_acc, global_step=epoch)
    file.write('Epoch: %d' % (epoch + 1))
    file.write(" | time in %d minutes, %d seconds\n" % (mins, secs))
    file.write(f'\tLoss: {I2P_train_loss:.4f}(train)\t|\tAcc: {I2P_train_acc * 100:.1f}%(train)\n')
    file.write(f'\tLoss: {I2P_valid_loss:.4f}(valid)\t|\tAcc: {I2P_valid_acc * 100:.1f}%(valid)\n')
    file.write("\n")
    file.flush()
file.close()
writer.close()

file = open(log_path + "/" + "high.txt", "w")
file.write(f'\tI2P Acc: {I2P_high_score * 100:.1f}%(valid)\n')
file.close()
