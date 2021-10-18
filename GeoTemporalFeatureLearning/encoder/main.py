from __future__ import print_function
import argparse
from math import  ceil
import random, shutil, json
from os.path import join, exists, isfile
from os import makedirs

import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from datetime import datetime
import torchvision.models as models
import opt as opts

from tensorboardX import SummaryWriter

import numpy as np
import evaluate


parser = argparse.ArgumentParser(description='encoder-NetVlad')
parser.add_argument('--mode', type=str, default='train', help='Mode', choices=['train'])
parser.add_argument('--batchSize', type=int, default=4, 
        help='Number of triplets (query, pos, negs). Each triplet consists of 12 images.')
parser.add_argument('--cacheBatchSize', type=int, default=24, help='Batch size for caching and testing')
parser.add_argument('--cacheRefreshRate', type=int, default=1000, 
        help='How often to refresh cache, in number of queries. 0 for off')
parser.add_argument('--nEpochs', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', 
        help='manual epoch number (useful on restarts)')
parser.add_argument('--nGPU', type=int, default=1, help='number of GPU to use.')
parser.add_argument('--optim', type=str, default='SGD', help='optimizer to use', choices=['SGD', 'ADAM'])
parser.add_argument('--lr', type=float, default=0.0001, help='Learning Rate.')
parser.add_argument('--lrStep', type=float, default=5, help='Decay LR ever N steps.')
parser.add_argument('--lrGamma', type=float, default=0.5, help='Multiply LR by Gamma for decaying.')
parser.add_argument('--weightDecay', type=float, default=0.001, help='Weight decay for SGD.')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD.')
parser.add_argument('--nocuda', action='store_true', help='Dont use cuda')
parser.add_argument('--threads', type=int, default=8, help='Number of threads for each data loader to use')
parser.add_argument('--seed', type=int, default=123, help='Random seed to use.')
parser.add_argument('--dataPath', type=str, default='./data/', help='Path for centroid data.')
parser.add_argument('--runsPath', type=str, default='./runs/', help='Path to save runs to.')
parser.add_argument('--savePath', type=str, default='checkpoints', 
        help='Path to save checkpoints to in logdir. Default=checkpoints/')
parser.add_argument('--cachePath', type=str, default='./', help='Path to save cache to.')

parser.add_argument('--resume', type=str, default='./runs/', help='Path to load checkpoint from, for resuming training or testing.')
parser.add_argument('--ckpt', type=str, default='best',
        help='Resume from latest or best checkpoint.', choices=['latest', 'best'])
parser.add_argument('--evalEvery', type=int, default=1, 
        help='Do a validation set run, and save, every N epochs.')
parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping. 0 is off.')
parser.add_argument('--dataset', type=str, default='bdd',
        help='Dataset to use', choices=['bdd'])
parser.add_argument('--arch', type=str, default='vgg16', 
        help='basenetwork to use', choices=['vgg16', 'alexnet'])
parser.add_argument('--vladv2', action='store_true', help='Use VLAD v2')
parser.add_argument('--pooling', type=str, default='netvlad', help='type of pooling to use',
        choices=['netvlad', 'max', 'avg'])
parser.add_argument('--num_clusters', type=int, default=64, help='Number of NetVlad clusters. Default=64')
parser.add_argument('--margin', type=float, default=4, help='Margin for triplet loss. Default=0.1')
parser.add_argument('--split', type=str, default='val', help='Data split to use for testing. Default is val', 
        choices=['test', 'test250k', 'train', 'val'])
parser.add_argument('--fromscratch', action='store_true', help='Train from scratch rather than using pretrained models')

class AverageMeter(object):
    """Computes and stores the average and current value"""
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

def validate_feats(model, val_loader, val_arg):
    print("Testing: ")
    model.eval()
    feat_size = 32768
    query_feats = np.zeros([len(val_loader) * val_arg.test_num_frames * val_arg.test_batch_size, feat_size])
    gallery_feats = np.zeros([len(val_loader) * val_arg.test_num_frames * val_arg.test_batch_size, feat_size])
    gallery_lat = np.zeros([len(val_loader) * val_arg.test_num_frames * val_arg.test_batch_size])
    gallery_lon = np.zeros([len(val_loader) * val_arg.test_num_frames * val_arg.test_batch_size])


    with torch.no_grad():
        for i, (bdd, gsv, dir_name, lat, lon) in enumerate(val_loader):
            bdd_height, bdd_width  = bdd.size(3), bdd.size(4)
            bdd = bdd.view(-1, 3, bdd_height, bdd_width).cuda()
            gsv_height, gsv_width = gsv.size(3), gsv.size(4)
            gsv = gsv.view(-1, 3, gsv_height, gsv_width).cuda()

            if i%200 == 0:
                print(i)

            image_encoding = model.encoder(bdd)
            feats_bdd = model.pool(image_encoding)
            image_encoding = model.encoder(gsv)
            feats_gsv = model.pool(image_encoding)

            query_feats[i * val_arg.test_num_frames:(i + 1) * val_arg.test_num_frames, :] = feats_bdd.view(-1, feat_size).cpu().detach().numpy()
            gallery_feats[i * val_arg.test_num_frames:(i + 1) * val_arg.test_num_frames, :] = feats_gsv.view(-1, feat_size).cpu().detach().numpy()

            gallery_lat[i * val_arg.test_num_frames:(i + 1) * val_arg.test_num_frames] = lat
            gallery_lon[i * val_arg.test_num_frames:(i + 1) * val_arg.test_num_frames] = lon



    dists_1 = 2 - 2 * np.matmul(query_feats, gallery_feats.transpose())
    idx_sorted_1 = np.argsort(dists_1, axis=1)
    top10_accuracy, top1_accuracy = evaluate.compute_accuracy(dists_1)
    error_dists = evaluate.compute_error_distance(gallery_lat, gallery_lon, gallery_lat, gallery_lon, idx_sorted_1)
    avg_error =  sum(error_dists) / len(error_dists)
    print("Testing:: Average error is {}\n\n\n".format( avg_error))

    return  avg_error, top10_accuracy, top1_accuracy


def train_model(model, optimizer, train_loader, criterion, epoch):
    losses = AverageMeter()
    print("\nTraining: ")
    model.train()

    start = time.time()
    temp = start


    regressor_loss = 0.0
    print_every = 10

    for ii, (bdd, gsv, dir_name, lat, lon) in enumerate(train_loader):
        torch.cuda.empty_cache()
        bdd_height, bdd_width = bdd.size(3), bdd.size(4)
        bdd = bdd.view(-1, 3, bdd_height, bdd_width).cuda()
        gsv_height, gsv_width = gsv.size(3), gsv.size(4)
        gsv = gsv.view(-1, 3, gsv_height, gsv_width).cuda()


        image_encoding = model.encoder(bdd)
        feats_bdd = model.pool(image_encoding)
        image_encoding = model.encoder(gsv)
        feats_gsv = model.pool(image_encoding)

        feats_gsv_neg = torch.flip(feats_gsv, [0])
        feats_loss = criterion(feats_bdd, feats_gsv, feats_gsv_neg)

        loss = 10*feats_loss
        losses.update(feats_loss, bdd.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()

        if (ii + 1) % 10 == 0:
            print("time = %dm, epoch %d, iter = %d, loss = %.3f, feats loss = %.3f, regressor loss = %.3f, avg feat loss = %.3f,  % ds  per % d   iters "
                  % ((time.time() - start) // 60, epoch + 1, ii + 1, loss, feats_loss, regressor_loss, losses.avg, time.time() - temp, print_every))

        temp = time.time()

    print("time = %dm, epoch %d, iter = %d, loss = %.3f, feats loss = %.3f, regressor loss = %.3f, avg feat loss = %.3f,  % ds  per % d   iters "
             % ((time.time() - start) // 60, epoch + 1, ii + 1, loss, feats_loss, regressor_loss, losses.avg, time.time() - temp, print_every))
    return  model


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    model_out_path = join(opt.savePath, filename)
    torch.save(state, model_out_path)
    if is_best:
        shutil.copyfile(model_out_path, join(opt.savePath, 'model_best.pth.tar'))

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, input):
        return F.normalize(input, p=2, dim=self.dim)

if __name__ == "__main__":
    opt = parser.parse_args()
    print(opt)
    restore_var = ['lr', 'lrStep', 'lrGamma', 'weightDecay', 'momentum', 
            'arch', 'num_clusters', 'pooling', 'optim',
             'seed', 'patience']
    if opt.resume:
        flag_file = join('checkpoints', 'flags.json')
        if exists(flag_file):
            with open(flag_file, 'r') as f:
                stored_flags = {'--'+k : str(v) for k,v in json.load(f).items() if k in restore_var}
                to_del = []
                for flag, val in stored_flags.items():
                    for act in parser._actions:
                        if act.dest == flag[2:]:
                            # store_true / store_false args don't accept arguments, filter these 
                            if type(act.const) == type(True):
                                if val == str(act.default):
                                    to_del.append(flag)
                                else:
                                    stored_flags[flag] = ''
                for flag in to_del: del stored_flags[flag]

                train_flags = [x for x in list(sum(stored_flags.items(), tuple())) if len(x) > 0]
                print('Restored flags:', train_flags)
                opt = parser.parse_args(train_flags, namespace=opt)

    print(opt)

    if opt.dataset.lower() == 'bdd':
        from bdd import BDD_GSV_Loader
    else:
        raise Exception('Unknown dataset')

    cuda = not opt.nocuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run with --nocuda")

    device = torch.device("cuda" if cuda else "cpu")

    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)

    print('===> Loading dataset(s)')

    train_arg_SF = opts.Arguments_Data_SF(phase='train')
    train_dataset_SF = BDD_GSV_Loader(train_arg_SF, flag='train')
    data_loader_train_SF = torch.utils.data.DataLoader(train_dataset_SF, batch_size=train_arg_SF.batch_size,
													 shuffle=True,
													 num_workers=0, drop_last=True)

    print(" Data available for SF train ", len(train_dataset_SF.trajectories))

    val_arg_SF = opts.Arguments_Data_SF(phase='val')
    val_dataset_SF = BDD_GSV_Loader(val_arg_SF)
    data_loader_val_SF = torch.utils.data.DataLoader(val_dataset_SF, batch_size=val_arg_SF.test_batch_size, shuffle=False,
											  num_workers=0, drop_last=True)

    print(" Data available for SF val ", len(val_dataset_SF.trajectories))

    val_arg_NY = opts.Arguments_Data_NY(phase='val')
    val_dataset_NY = BDD_GSV_Loader(val_arg_NY)
    data_loader_val_NY = torch.utils.data.DataLoader(val_dataset_NY, batch_size=val_arg_NY.test_batch_size, shuffle=False,
                                              num_workers=0, drop_last=True)

    print(" Data available for NY val ", len(val_dataset_NY.trajectories))

    val_arg_Berkeley = opts.Arguments_Data_Berkeley(phase='val')
    val_dataset_Berkeley = BDD_GSV_Loader(val_arg_Berkeley)
    data_loader_val_Berkeley = torch.utils.data.DataLoader(val_dataset_Berkeley, batch_size=val_arg_Berkeley.test_batch_size, shuffle=False,
                                              num_workers=0, drop_last=True)

    print(" Data available for Berkeley val ", len(val_dataset_Berkeley.trajectories))

    val_arg_BayArea = opts.Arguments_Data_BayArea(phase='train')
    val_dataset_BayArea = BDD_GSV_Loader(val_arg_BayArea)
    data_loader_val_BayArea = torch.utils.data.DataLoader(val_dataset_BayArea, batch_size=val_arg_BayArea.test_batch_size, shuffle=False,
                                              num_workers=0, drop_last=True)

    print(" Data available for BayArea val ", len(val_dataset_BayArea.trajectories))

    val_loader = [data_loader_val_SF, data_loader_val_NY, data_loader_val_BayArea, data_loader_val_Berkeley]
    val_args = [val_arg_SF, val_arg_NY, val_arg_BayArea, val_arg_Berkeley]
    worst_error = 10000000

    print('===> Building model')

    pretrained = not opt.fromscratch
    if opt.arch.lower() == 'alexnet':
        encoder_dim = 256
        encoder = models.alexnet(pretrained=pretrained)
        # capture only features and remove last relu and maxpool
        layers = list(encoder.features.children())[:-2]

        if pretrained:
            # if using pretrained only train conv5
            for l in layers[:-1]:
                for p in l.parameters():
                    p.requires_grad = False

    elif opt.arch.lower() == 'vgg16':
        encoder_dim = 512
        encoder = models.vgg16(pretrained=pretrained)
        # capture only feature part and remove last relu and maxpool
        layers = list(encoder.features.children())[:-2]

        if pretrained:
            # if using pretrained then only train conv5_1, conv5_2, and conv5_3
            for l in layers[:-5]: 
                for p in l.parameters():
                    p.requires_grad = False

    if opt.mode.lower() == 'cluster' and not opt.vladv2:
        layers.append(L2Norm())

    encoder = nn.Sequential(*layers)
    model = nn.Module() 
    model.add_module('encoder', encoder)

    isParallel = False
    if not opt.resume:
        model = model.to(device)
    
    if opt.mode.lower() == 'train':
        if opt.optim.upper() == 'ADAM':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, 
                model.parameters()), lr=opt.lr)#, betas=(0,0.9))
        elif opt.optim.upper() == 'SGD':
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, 
                model.parameters()), lr=opt.lr,
                momentum=opt.momentum,
                weight_decay=opt.weightDecay)

            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=opt.lrStep, gamma=opt.lrGamma)
        else:
            raise ValueError('Unknown optimizer: ' + opt.optim)

        criterion = nn.TripletMarginLoss(margin=opt.margin**0.5,
                p=2, reduction='sum').to(device)

    if opt.resume:
        if opt.ckpt.lower() == 'latest':
            resume_ckpt = join(opt.resume, 'checkpoints', 'checkpoint.pth.tar')
        elif opt.ckpt.lower() == 'best':
            resume_ckpt = join(opt.resume, 'checkpoints', 'model_best.pth.tar')

        if isfile(resume_ckpt):
            print("=> loading checkpoint '{}'".format(resume_ckpt))
            checkpoint = torch.load(resume_ckpt, map_location=lambda storage, loc: storage)
            opt.start_epoch = checkpoint['epoch']
            worst_error = checkpoint['localization_error']
            model.load_state_dict(checkpoint['state_dict'])
            model = model.to(device)
            if opt.mode == 'train':
                optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(resume_ckpt, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(resume_ckpt))


    if opt.mode.lower() == 'train':
        print('===> Training model')
        writer = SummaryWriter(log_dir=join(opt.runsPath, datetime.now().strftime('%b%d')+'_'+opt.arch+'_'+opt.pooling))

        # write checkpoints in logdir
        logdir = writer.file_writer.get_logdir()
        opt.savePath = join(logdir, opt.savePath)
        if not opt.resume:
            makedirs(opt.savePath)

        not_improved = 0
        best_score = 0

        for epoch in range(opt.start_epoch+1, opt.nEpochs + 1):
            if opt.optim.upper() == 'SGD':
                scheduler.step(epoch)
            train_model(model, optimizer, data_loader_train_SF, criterion, epoch)
            if (epoch % opt.evalEvery) == 0:
                localization_error = []
                for i in range(len(val_loader)):
                    current_error, top10_accuracy, top1_accuracy = validate_feats(model, val_loader[i], val_args[i])
                    localization_error.append(current_error)

                current_epoch_error = sum(localization_error)/len(localization_error)
                print("Current average error: ", current_epoch_error)

                is_best = current_epoch_error < worst_error
                if is_best:
                    worst_error = current_epoch_error

                    save_checkpoint({
                            'epoch': epoch,
                            'state_dict': model.state_dict(),
                            'localization_error': current_epoch_error,
                            'optimizer' : optimizer.state_dict(),
                            'parallel' : isParallel,
                    }, is_best)

        writer.close()
