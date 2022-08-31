# coding=utf-8
import os
import argparse
import math
import torch.nn.functional as F
from torch.utils.data import DataLoader,random_split
import torch.optim as optim
from models import *
from load_feature import *
from tqdm import tqdm
from evaluation import evaluate
from utils import save_args


# Training settings
parser = argparse.ArgumentParser()

# data files
parser.add_argument('--inputpath', type=str, default="/home/data/toy/bert_feature/",
                    help='input data path')
parser.add_argument('--outputresult', type=str, default="./result/exp/",
                    help='output file path')

# train
parser.add_argument('--epochs', type=int, default=30, help='Number of epochs to train.')
parser.add_argument('--batch_size', type=int, default=100, help='Batch size in train and test.')
parser.add_argument('--num_data', type=int, default=1000000, help='maximum number of data samples to use.')
parser.add_argument('--lr', default=5e-5, type=float, help='learning rate')
parser.add_argument('--lr_warm_up_num', default=1000, type=int, help='number of warm-up steps of learning rate')
parser.add_argument('--beta1', default=0.9, type=float, help='beta 1')
parser.add_argument('--beta2', default=0.999, type=float, help='beta 2')
parser.add_argument('--no_grad_clip', default=False, action='store_true', help='whether use gradient clip')
parser.add_argument('--max_grad_norm', default=5.0, type=float, help='global Norm gradient clipping rate')
parser.add_argument('--use_ema', default=False, action='store_true', help='whether use exponential moving average')
parser.add_argument('--ema_decay', default=0.9999, type=float, help='exponential moving average decay')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')

# model
parser.add_argument('--dropout', type=float, default=0.0, help='layers dropout rate in GRU.')
parser.add_argument('--GRU_layers', type=int, default=1, help='number of GRU layers.')
parser.add_argument('--bidirectional', type=bool, default=False, help='number of GRU directions [1,2].')

args = parser.parse_args()
save_args(args)

#使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_cuda = torch.cuda.is_available()
# #使用CPU
# device="cpu"
# use_cuda =False


# data
print("begin loading data............")
dataset = MyDataset(args.inputpath)
dataset_test,dataset_train = random_split(dataset, [int(len(dataset)*0.3), len(dataset)-int(len(dataset)*0.3)], generator=torch.Generator().manual_seed(0))
dataloader_train = DataLoader(dataset=dataset_train,shuffle=True,batch_size=args.batch_size)
dataloader_test= DataLoader(dataset=dataset_test,shuffle=False,batch_size=args.batch_size)

model = my_gru(args)

# optimizer and scheduler
parameters = filter(lambda p: p.requires_grad, model.parameters())
# optimizer = optim.SGD(parameters, lr=args.lr, momentum=0.9)
optimizer = optim.Adam(
    params=parameters,
    lr=args.lr,
    betas=(args.beta1, args.beta2),
    eps=1e-8,
    weight_decay=0)
cr = 1.0 / math.log(args.lr_warm_up_num)
scheduler = None
scheduler = optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=lambda ee: cr * math.log(ee + 1)
    if ee < args.lr_warm_up_num else 1)

torch.backends.cudnn.benchmark = True
if use_cuda:
    print("use cuda data")
    model.to(device)



def train_epoch(model, epoch, step, fout):
    loss_train = 0.0
    for i, (feature, label) in enumerate(tqdm((dataloader_train), desc='Train '+'Epoch-'+str(epoch+1))):
        if i >= args.num_data:
            break
        # get data
        feature = feature.to(device)
        label=label.to(device)
        # calculate loss and back propagation
        model.train()
        optimizer.zero_grad()
        output = model(feature)  
        loss = F.cross_entropy(output,label)
        loss_train += loss
        loss.backward()

        # gradient clip
        if (not args.no_grad_clip):
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        # update model
        optimizer.step()

        # update learning rate
        if scheduler is not None:
            scheduler.step()

        step += 1

    loss_train = loss_train / len(dataloader_train)

    # save checkpoint
    torch.save({
        'epoch': epoch+1,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict()},
        os.path.join(outpath, 'checkpoint.pth'))

    # print info
    print('Epoch: {:04d}'.format(epoch + 1),
          'loss_train: {:.4f}'.format(loss_train))
    fout.write('Epoch: ' + str(epoch + 1) + ',' +
               'loss_train: ' + str(loss_train) + ',' +
                '\n')

    return step


def train(model, args, fout):
    # resume checkpoint if file exists
    if os.path.isfile(os.path.join(outpath, 'checkpoint.pth')):
        print('resuming checkpoint experiment')
        checkpoint = torch.load(os.path.join(outpath, 'checkpoint.pth'), map_location=device)
    else:
        checkpoint = {
            'epoch': 0,
            'model': None,
            'optimizer': None,
            'scheduler': None}

    step = 0

    for epoch in range(checkpoint['epoch'],args.epochs):
        step = train_epoch(model, epoch, step, fout)
        with torch.no_grad():
            test(model, dataloader_test, epoch)

    # finalize model
    
    evaluate(model, dataloader_test,device, fout)
    if not os.path.exists(os.path.join(outpath, 'model')):
        os.mkdir(os.path.join(outpath, 'model'))
    torch.save(model.state_dict(), os.path.join(outpath, 'model', 'model.pth'))
    os.remove(os.path.join(outpath, 'checkpoint.pth'))


def test(model,dataloader_test, epoch):
    model.eval()
    loss_test = 0.0
    for i, (feature, label) in enumerate(tqdm((dataloader_test),desc='Test '+'Epoch-'+str(epoch+1))):
        feature = feature.to(device)
        label = label.to(device)
        output = model(feature)
        loss = F.cross_entropy(output, label)
        loss_test += loss
    loss_test = loss_test / len(dataloader_test)
    print('Epoch: {:04d}'.format(epoch + 1),
          'loss_test {:.4f}'.format(loss_test))
    fout.write('Epoch: ' + str(epoch + 1) + ',' +
               'loss_test: ' + str(loss_test) + ',' +
                '\n')


if __name__ == '__main__':
    outpath = args.outputresult
    if not os.path.exists(outpath): os.mkdir(outpath) 
    fout = open(outpath+'/results.txt', 'a')
    train(model, args, fout)
    print("Optimization Finished!")
    fout.close()
