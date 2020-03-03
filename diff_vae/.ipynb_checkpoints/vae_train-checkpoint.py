# -*- coding: utf-8 -*- 

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from torch.autograd import Variable

import math, random, sys
import numpy as np
import argparse
from collections import deque
import cPickle as pickle

from fast_jtnn import *
import rdkit

from datetime import datetime
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import os # for save plot

def save_KL_plt(save_dir, epoch, x, kl):
    plt.plot(x, kl)
    plt.xlabel('Iteration')
    plt.ylabel('KL divergence')
    plt.grid()
    plt.savefig('./plot/{}/KL/epoch_{}.png'.format(str(save_dir),str(epoch)))
    plt.close()
def save_Acc_plt(save_dir, epoch, x, word, topo, assm):
    plt.plot(x, word)
    plt.plot(x, topo)
    plt.plot(x, assm)
    plt.xlabel('Iteration')
    plt.ylabel('Acc')
    plt.legend(['Word acc','Topo acc','Assm acc'])
    plt.grid()
    plt.savefig('./plot/{}/Acc/epoch_{}.png'.format(str(save_dir),str(epoch)))
    plt.close()
def save_Norm_plt(save_dir, epoch, x, pnorm, gnorm):
    plt.plot(x, pnorm)
    plt.plot(x, gnorm)
    plt.xlabel('Iteration')
    plt.ylabel('Norm')
    plt.legend(['Pnorm', 'Gnorm'])
    plt.grid()
    plt.savefig('./plot/{}/Norm/epoch_{}.png'.format(str(save_dir),str(epoch)))
    plt.close()
    
lg = rdkit.RDLogger.logger() 
lg.setLevel(rdkit.RDLogger.CRITICAL)

parser = argparse.ArgumentParser()
parser.add_argument('--train', required=True)
parser.add_argument('--vocab', required=True)
parser.add_argument('--save_dir', type=str, default=None)
parser.add_argument('--load_epoch', type=int, default=-1)

parser.add_argument('--hidden_size', type=int, default=300)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--rand_size', type=int, default=8)
parser.add_argument('--depthT', type=int, default=6)
parser.add_argument('--depthG', type=int, default=3)
parser.add_argument('--share_embedding', action='store_true')
parser.add_argument('--use_molatt', action='store_true')

parser.add_argument('--clip_norm', type=float, default=50.0)
parser.add_argument('--beta', type=float, default=1.0)
parser.add_argument('--epoch', type=int, default=20)
parser.add_argument('--anneal_rate', type=float, default=0.9)
parser.add_argument('--lr', type=float, default=1e-3)

args = parser.parse_args()
print args
  
vocab = [x.strip("\r\n ") for x in open(args.vocab)] 
vocab = Vocab(vocab)

model = DiffVAE(vocab, args).cuda()

for param in model.parameters():
    if param.dim() == 1:
        nn.init.constant_(param, 0)
    else:
        nn.init.xavier_normal_(param)

#기존 모델 가져와서 학습하려면 이거 쓰면될 듯
if args.load_epoch >= 0:
    model.load_state_dict(torch.load(args.save_dir + "/model.iter-" + str(args.load_epoch)))

print "Model #Params: %dK" % (sum([x.nelement() for x in model.parameters()]) / 1000,)

optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = lr_scheduler.ExponentialLR(optimizer, args.anneal_rate)
scheduler.step()

PRINT_ITER = 20
param_norm = lambda m: math.sqrt(sum([p.norm().item() ** 2 for p in m.parameters()]))
grad_norm = lambda m: math.sqrt(sum([p.grad.norm().item() ** 2 for p in m.parameters() if p.grad is not None]))

folder_name = str(datetime.now()) +'_'+ args.save_dir.split('/')[-1]
os.makedirs('./plot/'+folder_name+'/KL')
os.makedirs('./plot/'+folder_name+'/Acc')
os.makedirs('./plot/'+folder_name+'/Norm')
print("...Finish Making Plot Folder...")
#Plot
x_plot=[]
kl_plot=[]
word_plot=[]
topo_plot=[]
assm_plot=[]
pnorm_plot=[]
gnorm_plot=[]

for epoch in xrange(args.load_epoch + 1, args.epoch):    
    start = datetime.now()
    print("EPOCH: %d | TIME: %s " % (epoch+1, str(start)))
    
    loader = PairTreeFolder(args.train, vocab, args.batch_size, num_workers=4)
    meters = np.zeros(4)

    for it, batch in enumerate(loader):
        x_batch, y_batch = batch
        try:
            model.zero_grad()
            loss, kl_div, wacc, tacc, sacc = model(x_batch, y_batch, args.beta)
            loss.backward()
        except Exception as e:
            print e
            continue

        nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
        optimizer.step()

        meters = meters + np.array([kl_div, wacc * 100, tacc * 100, sacc * 100])
        '''
            KL_div: prior distribution p(z)와 Q(z|X,Y)와의 KL div
            Word: Label Prediction acc
            Topo: Topological Prediction acc
            Assm: 조립할 때, 정답과 똑같이 했는가? acc
        '''
        
        #Plot
#         x_plot.append(int(it))
#         kl_plot.append(kl_div)
#         word_plot.append(wacc * 100)
#         topo_plot.append(tacc * 100)
#         assm_plot.append(sacc * 100)
        
#         pnorm= param_norm(model)
#         pnorm_plot.append(pnorm)
#         gnorm = grad_norm(model)
#         gnorm_plot.append(gnorm)
        
        #print(it)
        if (it + 1) % PRINT_ITER == 0:
            meters /= PRINT_ITER
            
            pnorm= param_norm(model)
            gnorm = grad_norm(model)
            
            print "KL: %.2f, Word: %.2f, Topo: %.2f, Assm: %.2f, PNorm: %.2f, GNorm: %.2f, iter: %d " % (meters[0], meters[1], meters[2], meters[3], pnorm, gnorm, it+1)
            
            x_plot.append(it+1)
            kl_plot.append(meters[0])
            word_plot.append(meters[1])
            topo_plot.append(meters[2])
            assm_plot.append(meters[3])
            pnorm_plot.append(pnorm)
            gnorm_plot.append(gnorm)
            
            sys.stdout.flush()
            meters *= 0
#         if (it+1) == 40:
#             break
            
    #Plot per 1 epoch
    print "Cosume Time per Epoch %s" % (str(datetime.now()-start))
    save_KL_plt(folder_name, epoch, x_plot, kl_plot)
    save_Acc_plt(folder_name, epoch, x_plot, word_plot, topo_plot, assm_plot)
    save_Norm_plt(folder_name, epoch, x_plot, pnorm_plot, gnorm_plot)
    x_plot=[]
    kl_plot=[]
    word_plot=[]
    topo_plot=[]
    assm_plot=[]
    pnorm_plot=[]
    gnorm_plot=[]
    
    scheduler.step()
    
    print "learning rate: %.6f" % scheduler.get_lr()[0]
    if args.save_dir is not None:
        torch.save(model.state_dict(), args.save_dir + "/model.iter-" + str(epoch))

