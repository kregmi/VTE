from __future__ import print_function
import argparse
import random, shutil, json
from os.path import join, exists, isfile
from os import makedirs
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as models
import numpy as np
import opt as opts

parser = argparse.ArgumentParser(description='encoder-NetVlad')
parser.add_argument('--mode', type=str, default='test', help='Mode', choices=['train'])
parser.add_argument('--batchSize', type=int, default=1,
					help='Number of triplets (query, pos, negs). Each triplet consists of 12 images.')
parser.add_argument('--cacheBatchSize', type=int, default=24, help='Batch size for caching and testing')
parser.add_argument('--cacheRefreshRate', type=int, default=1000,
					help='How often to refresh cache, in number of queries. 0 for off')
parser.add_argument('--nEpochs', type=int, default=30, help='number of epochs to train for')
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
parser.add_argument('--cachePath', type=str, default='./data/', help='Path to save cache to.')

parser.add_argument('--resume', type=str, default='./vgg16_netvlad_checkpoint',
					help='Path to load checkpoint from, for resuming training or testing.')
parser.add_argument('--ckpt', type=str, default='latest',
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
parser.add_argument('--margin', type=float, default=0.1, help='Margin for triplet loss. Default=0.1')
parser.add_argument('--split', type=str, default='val', help='Data split to use for testing. Default is val',
					choices=['test', 'test250k', 'train', 'val'])
parser.add_argument('--fromscratch', action='store_true', help='Train from scratch rather than using pretrained models')
parser.add_argument('--feats_save_dir', type=str, default='./NetVLAD_Feats/',
					help='Dataset to use', choices=['bdd'])
parser.add_argument('--gsv4imgs', action='store_true', help='Save features for all 4 GSV images')

import scipy.io as sio


def save_features(test_data_loader):
	num_gsv_frames = 4
	torch.cuda.empty_cache()
	model.eval()
	opt.feats_save_dir = join(opt.feats_save_dir, opt.phase)
	if not exists(opt.feats_save_dir):
		makedirs(opt.feats_save_dir)



	with torch.no_grad():
		print('====> Extracting Features')
		pool_size = encoder_dim
		if opt.pooling.lower() == 'netvlad': pool_size *= opt.num_clusters
		vlad_gsv = np.empty((num_gsv_frames*30, pool_size))
		# vlad_bdd = np.empty((30, pool_size))

		for iteration, (bdd, gsv, dir_name, lat, lon) in enumerate(test_data_loader):
			torch.cuda.empty_cache()
			print("Iteration : " + str(iteration ) +  '  ' + dir_name[0])
			bdd = bdd.view(bdd.shape[2], -1, bdd.shape[3], bdd.shape[4]).to(device)
			gsv = gsv.view(gsv.shape[2], -1, gsv.shape[3], gsv.shape[4]).to(device)

			vlad_encoding_bdd = model.pool(model.encoder(bdd))
			vlad_bdd = vlad_encoding_bdd.cpu().detach().numpy()
			torch.cuda.empty_cache()

			# if we want to save features for all 4 gsv images
			for i in range(4):
				input = gsv[i * 30:(i + 1) * 30, :, :, :]
				vlad_encoding_gsv = model.pool(model.encoder(input))
				vlad_gsv[i * 30:(i + 1) * 30, :] = vlad_encoding_gsv.cpu().detach().numpy()

			torch.cuda.empty_cache()


			matfilename = join(opt.feats_save_dir, dir_name[0] + '.mat')
			sio.savemat(matfilename, dict([('lat', lat), ('lon', lon), ('bdd_feats', vlad_bdd), ('gsv_feats', vlad_gsv)]))
			torch.cuda.empty_cache()

	print("Done!!!")




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
				   'runsPath', 'savePath', 'arch', 'num_clusters', 'pooling', 'optim',
				   'margin', 'seed', 'patience']
	if opt.resume:
		flag_file = join(opt.resume, 'checkpoints', 'flags.json')
		if exists(flag_file):
			with open(flag_file, 'r') as f:
				stored_flags = {'--' + k: str(v) for k, v in json.load(f).items() if k in restore_var}
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
	opt.phase = 'val'
	# opt.phase = 'train'
	train_arg = opts.Arguments_Data(phase=opt.phase)
	train_arg.gsv4imgs = opt.gsv4imgs
	train_dataset = BDD_GSV_Loader(train_arg, gsv4imgs=train_arg.gsv4imgs)
	data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_arg.test_batch_size, shuffle=False,
											 num_workers=0, drop_last=True)

	print("Data available to work on right now ", len(train_dataset.trajectories))


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
	if opt.nGPU > 1 and torch.cuda.device_count() > 1:
		model.encoder = nn.DataParallel(model.encoder)
		if opt.mode.lower() != 'cluster':
			model.pool = nn.DataParallel(model.pool)
		isParallel = True

	if not opt.resume:
		model = model.to(device)

	if opt.mode.lower() == 'train':
		if opt.optim.upper() == 'ADAM':
			optimizer = optim.Adam(filter(lambda p: p.requires_grad,
										  model.parameters()), lr=opt.lr)  # , betas=(0,0.9))
		elif opt.optim.upper() == 'SGD':
			optimizer = optim.SGD(filter(lambda p: p.requires_grad,
										 model.parameters()), lr=opt.lr,
								  momentum=opt.momentum,
								  weight_decay=opt.weightDecay)

			scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=opt.lrStep, gamma=opt.lrGamma)
		else:
			raise ValueError('Unknown optimizer: ' + opt.optim)

		criterion = nn.TripletMarginLoss(margin=opt.margin ** 0.5,
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
			best_metric = checkpoint['best_score']
			model.load_state_dict(checkpoint['state_dict'])
			model = model.to(device)
			if opt.mode == 'train':
				optimizer.load_state_dict(checkpoint['optimizer'])
			print("=> loaded checkpoint '{}' (epoch {})"
				  .format(resume_ckpt, checkpoint['epoch']))
		else:
			print("=> no checkpoint found at '{}'".format(resume_ckpt))



	print('===> Running evaluation step')
	save_features(data_loader)
	print("Done !!!")
