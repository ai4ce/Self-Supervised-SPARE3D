from __future__ import print_function, division
import torch
import argparse
import os
from tensorboardX import SummaryWriter
import time
from model import SR
from Dataloader import SR_data,P2I_data,I2P_data
import json
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument('--Dataroot', default=None, required=True, help='path to dataset')
parser.add_argument('--Dataset', default=".", required=True,
                    help='path to validating dataset')
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
model = SR(opt.pretrained).to(opt.device)
dataroot = opt.Dataroot
with open(opt.Dataset, 'r') as f:
    database = json.load(f)
dataset_train = database.get('I2P')
dataset_train = dataset_train.get('train')
dataset_train_ = {}
for i in list(dataset_train.keys())[:opt.num_train]:
    dataset_train_.update({i: dataset_train[i]})
dataset_train = dataset_train_

dataset_valid_P2I = database.get('P2I')
dataset_valid_P2I = dataset_valid_P2I.get('valid')

dataset_valid_I2P = database.get('I2P')
dataset_valid_I2P = dataset_valid_I2P.get('valid')


def train_model():
    batch_loss = 0
    train_data = SR_data(os.path.join(dataroot, 'train'), dataset_train)
    data_train = torch.utils.data.DataLoader(train_data, batch_size=opt.batchSize, pin_memory=True, num_workers=4,
                                             shuffle=True)
    model.train()
    for i, (Front_img, Right_img, Top_img, pose) in enumerate(
            data_train):
        optimizer.zero_grad()
        Front_img, Right_img, Top_img= Front_img.to(device), Right_img.to(device), Top_img.to(device)
        pose=[p.to(device) for p in pose]
        y=[model(Front_img.float(), Right_img.float(), Top_img.float(), p.float()) for p in pose]
        Y = torch.cat([torch.unsqueeze(j, 1) for j in y], axis=1)
        Y=F.normalize(Y,dim=2)
        Y_hat=torch.eye(8)
        Y_hat=Y_hat.expand(Front_img.shape[0],8,8)
        Y_hat=Y_hat.to(device)
        loss = criterion(Y, Y_hat)
        batch_loss += loss.item() * Front_img.shape[0]
        loss.backward()
        optimizer.step()
    epoch_loss = batch_loss / len(train_data)
    return epoch_loss

def Eval_I2P():
    eval_loss = 0
    eval_acc = 0
    eval_data = I2P_data(os.path.join(dataroot,'valid'),dataset_valid_I2P)
    data_eval = torch.utils.data.DataLoader(eval_data,batch_size=opt.batchSize,pin_memory=True,num_workers=4, shuffle=True)
    with torch.no_grad():
        model.eval()
        for i, (Inputf,Inputr,Inputt,Inputa,Label) in enumerate(data_eval):
            Inputf,Inputr,Inputt,Inputa,Label=Inputf.to(device),Inputr.to(device),Inputt.to(device),Inputa.to(device),Label.to(device)
            y = F.normalize(model(Inputf.float(),Inputr.float(),Inputt.float(),Inputa.float()),dim=1)
            y=torch.cat((y[:,:2],y[:,4:6]),axis=1)
            Label=Label.reshape(Inputf.shape[0])
            loss = criterion_I2P(y, Label)
            eval_loss += loss.item()* Inputf.shape[0]
            eval_acc += (y.argmax(1) == Label).sum().item()
        epoch_eval_loss = eval_loss / len(eval_data)
        epoch_eval_acc =eval_acc / len(eval_data)
    return epoch_eval_loss, epoch_eval_acc

def Eval_P2I():
    eval_loss = 0
    eval_acc = 0
    eval_data = P2I_data(os.path.join(dataroot, 'valid'), dataset_valid_P2I)
    data_eval = torch.utils.data.DataLoader(eval_data, batch_size=opt.batchSize, pin_memory=True, num_workers=4,
                                            shuffle=True)
    with torch.no_grad():
        model.eval()
        for i, (Front_img, Right_img, Top_img, Ans_1, Ans_2, Ans_3, Ans_4, Label, View_vector) in enumerate(data_eval):
            Front_img, Right_img, Top_img, Ans_1, Ans_2, Ans_3, Ans_4, Label, View_vector = Front_img.to(
                device), Right_img.to(device), Top_img.to(device), Ans_1.to(device), Ans_2.to(device), Ans_3.to(
                device), Ans_4.to(device), Label.to(device), View_vector.to(device)
            y_1 = F.normalize(model(Front_img.float(), Right_img.float(), Top_img.float(), Ans_1.float()),dim=1)
            y_2 = F.normalize(model(Front_img.float(), Right_img.float(), Top_img.float(), Ans_2.float()),dim=1)
            y_3 = F.normalize(model(Front_img.float(), Right_img.float(), Top_img.float(), Ans_3.float()),dim=1)
            y_4 = F.normalize(model(Front_img.float(), Right_img.float(), Top_img.float(), Ans_4.float()),dim=1)
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
criterion_I2P = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
log_path = opt.outf
writer = SummaryWriter(opt.outf)

if os.path.exists(log_path) == False:
    os.makedirs(log_path)

file = open(log_path + "/SR_Lr_" + str(opt.lr) + ".txt", "w")
P2I_high_score = 0
I2P_high_score = 0

for epoch in range(N_EPOCHS):
    start_time = time.time()
    train_loss= train_model()

    P2I_valid_loss, P2I_valid_acc = Eval_P2I()
    I2P_valid_loss, I2P_valid_acc = Eval_I2P()
    print(train_loss,I2P_valid_acc,P2I_valid_acc)
    if P2I_high_score < P2I_valid_acc:
        P2I_high_score = P2I_valid_acc
        torch.save(model.state_dict(), log_path + "/P2I_Lr_" + str(opt.lr) + ".pth")
    if I2P_high_score < I2P_valid_acc:
        I2P_high_score = I2P_valid_acc
        torch.save(model.state_dict(), log_path + "/I2P_Lr_" + str(opt.lr) + ".pth")
    secs = int(time.time() - start_time)
    mins = secs / 60
    secs = secs % 60
    writer.add_scalar('train_loss', train_loss, global_step=epoch)
    writer.add_scalar('P2I_acc', P2I_valid_acc, global_step=epoch)
    writer.add_scalar('I2P_acc', I2P_valid_acc, global_step=epoch)
    file.write('Epoch: %d' % (epoch + 1))
    file.write(" | time in %d minutes, %d seconds\n" % (mins, secs))
    file.write(f'\tSR training Loss: {train_loss:.4f}(train)\n')
    file.write(f'\tI2P valid acc: {I2P_valid_acc* 100:.1f}(valid)\t|\tP2I valid acc: {P2I_valid_acc* 100:.1f}(valid)\n')
    file.write("\n")
    file.flush()
file.close()
writer.close()

file = open(log_path + "/" + "high.txt", "w")
file.write(f'\tI2P Acc: {I2P_high_score * 100:.1f}%(valid)\tP2I Acc: {P2I_high_score * 100:.1f}%(valid)\n')
file.close()
