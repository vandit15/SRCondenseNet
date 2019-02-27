from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import argparse
import os
import shutil
import time
import random
import math
import warnings
import models
from utils import convert_model, measure_model, charbonnier_loss, TrainDatasetFromFolder, read_data, testDatasetFromFolder2, compute_ssim
from torch.utils import data
import torch
import cv2

parser = argparse.ArgumentParser(description='PyTorch Condensed Convolutional Networks')
parser.add_argument('--train_data', metavar='DIR',
                    help='path to train dataset')
parser.add_argument('--test_data', metavar='DIR',
                    help='path to testidation dataset')
parser.add_argument('--model', default='condensenet', type=str, metavar='M',
                    help='model to train the dataset')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=120, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch_size', default=64, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=1e-5, type=float,
                    metavar='LR', help='initial learning rate (default: 0.1)')
parser.add_argument('--dropout_rate', '--dropout_rate', default=0.25, type=float,
                    metavar='DR')
parser.add_argument('--lr-type', default='cosine', type=str, metavar='T',
                    help='learning rate strategy (default: cosine)',
                    choices=['cosine', 'multistep'])
parser.add_argument('--weight-decay', '--wd', default=1e-5, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=750, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model (default: false)')
parser.add_argument('--no-save-model', dest='no_save_model', action='store_true',
                    help='only save best model (default: false)')
parser.add_argument('--manual-seed', default=0, type=int, metavar='N',
                    help='manual seed (default: 0)')
parser.add_argument('--gpu',
                    help='gpu available')
parser.add_argument('--savedir', type=str, metavar='PATH', default='results/savedir',
                    help='path to save result and checkpoint (default: results/savedir)')
parser.add_argument('--resume', action='store_true',
                    help='use latest checkpoint if have any (default: none)')

parser.add_argument('--stages', type=str, metavar='STAGE DEPTH',
                    help='per layer depth')
parser.add_argument('--bottleneck', default=12, type=int, metavar='B',
                    help='bottleneck (default: 4)')
parser.add_argument('--group-1x1', type=int, metavar='G', default=4,
                    help='1x1 group convolution (default: 4)')
parser.add_argument('--group-3x3', type=int, metavar='G', default=4,
                    help='3x3 group convolution (default: 4)')
parser.add_argument('--condense-factor', type=int, metavar='C', default=4,
                    help='condense factor (default: 4)')
parser.add_argument('--growth', type=str, metavar='GROWTH RATE',
                    help='per layer growth')
parser.add_argument('--group-lasso-lambda', default=0., type=float, metavar='LASSO',
                    help='group lasso loss weight (default: 0)')

parser.add_argument('--evaluate', action='store_true',
                    help='evaluate model on test set (default: false)')
parser.add_argument('--convert-from', default=None, type=str, metavar='PATH',
                    help='path to saved checkpoint (default: none)')
parser.add_argument('--evaluate-from', default=None, type=str, metavar='PATH',
                    help='path to saved checkpoint (default: none)')
parser.add_argument('--scaling_factor', default=2, type = int,choices = [2,4,8])
parser.add_argument('--inp_img_size', default=64, type = int)
parser.add_argument('--c_dim', default=1, type=int)
parser.add_argument('--clip', default=0.4, type=float)


args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
args.stages = list(map(int, args.stages.split('-')))
args.growth = list(map(int, args.growth.split('-')))
if args.condense_factor is None:
    args.condense_factor = args.group_1x1

warnings.filterwarnings("ignore")


import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np

torch.manual_seed(args.manual_seed)
torch.cuda.manual_seed_all(args.manual_seed)
test_batch_size = 1
best_psnr = 0

class TensorClass(data.TensorDataset):
    def __init__(self, data_tensor, target_tensor, bicubic_tensor):
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor
        self.bicubic_tensor = bicubic_tensor

    def __getitem__(self, index):
        return self.data_tensor[index], self.target_tensor[index], self.bicubic_tensor[index]

    def __len__(self):
        return self.data_tensor.size(0)

def main():
    global args, model

    ### Calculate FLOPs & Param
    model = getattr(models, args.model)(args)
    IMAGE_SIZE = args.inp_img_size
    n_flops, n_params = measure_model(model.cuda(), IMAGE_SIZE, IMAGE_SIZE, args.scaling_factor)
    print('FLOPs: %.2fM, Params: %.2fM' % (n_flops / 1e6, n_params / 1e6))
    args.filename = "%s_%s_%s.txt" % \
        (args.model, int(n_params), int(n_flops))
    del(model)

    ###########  Building Model ##############
    cudnn.benchmark = True
    model = getattr(models, args.model)(args)
    model = torch.nn.DataParallel(model).cuda()
    criterion = charbonnier_loss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    lr = args.lr

    ##########  Data Loading ###############
    traindir = os.path.join("../data/", args.train_data)
    testdir = os.path.join("../data/", args.test_data)
    
    train_img = read_data(traindir, crop_size=args.inp_img_size, upscale_factor=args.scaling_factor, c_dim=args.c_dim, stride = 128)
    train_set = TensorClass(train_img[0], train_img[1], train_img[2])
    test_set = testDatasetFromFolder2(testdir, upscale_factor=args.scaling_factor)
    train_loader = data.DataLoader(dataset=train_set, num_workers=args.workers, batch_size=args.batch_size, shuffle=True)
    test_loader = data.DataLoader(dataset=test_set, num_workers=args.workers, batch_size=test_batch_size, shuffle=False)
    
    

    ### Optionally resume from a checkpoint
    if args.resume:
        checkpoint = load_checkpoint(args)
        if checkpoint is not None:
            args.start_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])

    ### Optionally evaluate from a model
    if args.evaluate_from is not None:
        args.evaluate = True
        state_dict = torch.load(args.evaluate_from)['state_dict']
        model.load_state_dict(state_dict)
    
    if args.evaluate:
        test_psnr, test_ssim = test(test_loader, model, criterion)
        print ("test Psnr: " + str(test_psnr) + "\t test SSIM: " + str(test_ssim) )
        return

    for epoch in range(args.start_epoch, args.epochs):
        ### Train for one epoch
        tr_psnr, tr_ssim ,loss, lr = \
            train(train_loader, model, criterion, optimizer, epoch)
        ### Evaluate on test set
        test_psnr, test_ssim = test(test_loader, model, criterion)
        
        print ("Train Psnr: " + str(tr_psnr) + "\t Train SSIM: " + str(tr_ssim) + "\t Loss: " + str(loss))
        print ("test Psnr: " + str(test_psnr) + "\t test SSIM: " + str(test_ssim))
        
        ### save checkpoint
        model_filename = 'checkpoint_%03d.pth.tar' % epoch
        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'psnr': best_psnr,
        }, args, model_filename)
    


### PSNR func
def psnr(imgs1, imgs2):
    sum = 0.0
    for i in range(len(imgs1)):
        mse = np.mean( (imgs1[i] - imgs2[i]) ** 2 )
        if mse == 0:
            return 100
        PIXEL_MAX = 1.0
        sum +=  20 * math.log10(PIXEL_MAX / math.sqrt(mse))
    return sum



def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    learned_module_list = []

    ### Switch to train mode
    model.train()
    
    ### Find all learned convs to prepare for group lasso loss
    for m in model.modules():
        if m.__str__().startswith('LearnedGroupConv'):
            learned_module_list.append(m)
    
    running_lr = None
    psnr_value = 0.0
    ssim_value = 0.0
    total_images = 0
    
    for i, (low_res_input, input, low_res_bicubic) in enumerate(train_loader):
        total_images += len(low_res_bicubic)
        progress = float(epoch * len(train_loader) + i) / \
            (args.epochs * len(train_loader))
        args.progress = progress
        
        ### Adjust learning rate
        # lr = adjust_learning_rate(optimizer, epoch, args, batch=i,
                                  # nBatch=len(train_loader), method=args.lr_type)
        lr = args.lr
        if running_lr is None:
            running_lr = lr
        input_var = torch.autograd.Variable(input.cuda(async=True))
        low_res_input_var = torch.autograd.Variable(low_res_input)
        low_res_bicubic = torch.autograd.Variable(low_res_bicubic)
        
        ### Compute output
        output = model(low_res_input_var, low_res_bicubic,progress)
        loss = criterion(output, input_var)
        
        ### Add group lasso loss
        if args.group_lasso_lambda > 0:
            lasso_loss = 0
            for m in learned_module_list:
                lasso_loss = lasso_loss + m.lasso_loss
            loss = loss + args.group_lasso_lambda * lasso_loss

        
        losses.update(loss.data[0], input.size(0))

        ### Compute gradient and do optimize step
        optimizer.zero_grad()
        loss.backward()
        # nn.utils.clip_grad_norm(model.parameters(),0.4)
        optimizer.step()
        ### Measure accuracy and record loss
        psnr_value += psnr(output.data.cpu().numpy().astype(np.float32), input_var.data.cpu().numpy().astype(np.float32))/len(low_res_bicubic)
        # for j in range(len(output.data)):
        #     ssim_value += compute_ssim(output.data[j].cpu(), input[j].cpu())
        

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Data {data_time.val:.3f}\t' 
                  'Loss {loss.avg:.4f}\t'  
                  'lr {lr: .4f}'.format(
                      epoch, i, len(train_loader),
                      data_time=data_time, loss=losses, lr=lr))
    return psnr_value/len(train_loader), ssim_value/total_images ,losses.avg, running_lr


def test(test_loader, model, criterion):
    losses = AverageMeter()

    ### Switch to evaluate mode
    model.eval()
    
    test_psnr = 0.0
    ssim_value = 0.0
    total_images = 0
    
    for i,(low_res_input, input, low_res_bicubic, ycrcb) in enumerate(test_loader):
        total_images += len(low_res_bicubic)

        input_var = torch.autograd.Variable(input.cuda(async=True))
        low_res_input_var = torch.autograd.Variable(low_res_input, volatile=True)
        low_res_bicubic = torch.autograd.Variable(low_res_bicubic, volatile=True)
        
        ### Compute output
        start = time.time()
        output = model(low_res_input_var, low_res_bicubic)
        end = time.time()
        batch_time = end - start
        loss = criterion(output, input_var)

        # t = test_psnr
        test_psnr += psnr(output.data.cpu().numpy().astype(np.float32), input_var.data.cpu().numpy().astype(np.float32))
        # print (test_psnr - t,)

        losses.update(loss.data[0], input.size(0))

        # for j in range(len(output.data)):
        #     ssim_value += compute_ssim(output.data[j].cpu(), input[j].cpu())
        for j in range(len(low_res_input)):
            cb, cr = ycrcb[j].numpy()[1], ycrcb[j].numpy()[2]
            bicubic_res = cv2.merge((low_res_bicubic[j][0].data.cpu().numpy(), cr, cb))
            bicubic_res = cv2.cvtColor(bicubic_res, cv2.COLOR_YCR_CB2BGR)
            cv2.imwrite('set14resultsx4/bicubic_res%d.jpg' %(i*test_batch_size + j+1), bicubic_res*255)
            out_res = cv2.merge((output.data[j][0].cpu().numpy(), cr, cb))
            out_res = cv2.cvtColor(out_res, cv2.COLOR_YCR_CB2BGR)
            cv2.imwrite('set14resultsx4/out_res%d.jpg' %(i*test_batch_size + j+1), out_res*255)

    return test_psnr/total_images, ssim_value/total_images

def load_checkpoint(args):
    model_dir = os.path.join(args.savedir, 'best_save_models4_3')
    latest_filename = os.path.join(model_dir, 'latest.txt')
    if os.path.exists(latest_filename):
        with open(latest_filename, 'r') as fin:
            model_filename = fin.readlines()[0]
    else:
        return None
    print("=> loading checkpoint '{}'".format(model_filename))
    state = torch.load(model_filename)
    print("=> loaded checkpoint '{}'".format(model_filename))
    return state


def save_checkpoint(state, args, filename):
    model_dir = os.path.join(args.savedir, 'best_save_models4_3')
    model_filename = os.path.join(model_dir, filename)
    latest_filename = os.path.join(model_dir, 'latest.txt')
    torch.save(state, model_filename)
    with open(latest_filename, 'w') as fout:
        fout.write(model_filename)

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, args, batch=None,
                         nBatch=None, method='cosine'):
    if method == 'cosine':
        T_total = args.epochs * nBatch
        T_cur = (epoch % args.epochs) * nBatch + batch
        lr = 0.5 * args.lr * (1 + math.cos(math.pi * T_cur / T_total))
    elif method == 'multistep':
        if args.data in ['cifar10', 'cifar100']:
            lr, decay_rate = args.lr, 0.1
            if epoch >= args.epochs * 0.75:
                lr *= decay_rate**2
            elif epoch >= args.epochs * 0.5:
                lr *= decay_rate
        else:
            """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
            lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr



if __name__ == '__main__':
    main()
