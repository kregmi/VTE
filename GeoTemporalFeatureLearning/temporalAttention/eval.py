from models import build_model
from data_loader_bdd_gsv_feats import BDD_GSV_EncoderFeats_Dataset
from losses import TripletLoss
from metrics import AccuracyMetric
from optimizers import NoamOptimizer
from evaluater import GTFL_Trainer

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
import numpy as np

from argparse import ArgumentParser
import json
import random


parser = ArgumentParser(description='Eval Transformer')
parser.add_argument('--config', type=str, default=None)

parser.add_argument('--data_dir', type=str, default='./data')
parser.add_argument('--save_config', type=str, default=None)
parser.add_argument('--save_checkpoint', type=str, default=None)
parser.add_argument('--save_log', type=str, default=None)

parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

parser.add_argument('--dataset_limit', type=int, default=None)
parser.add_argument('--print_every', type=int, default=1)
parser.add_argument('--save_every', type=int, default=1)

parser.add_argument('--vocabulary_size', type=int, default=None)
parser.add_argument('--positional_encoding', action='store_true')

parser.add_argument('--d_model', type=int, default=512)
parser.add_argument('--in_feat_size', type=int, default=32768)
parser.add_argument('--layers_count', type=int, default=1)
parser.add_argument('--heads_count', type=int, default=2)
parser.add_argument('--d_ff', type=int, default=128)
parser.add_argument('--dropout_prob', type=float, default=0.1)

parser.add_argument('--label_smoothing', type=float, default=0.1)
parser.add_argument('--optimizer', type=str, default="Adam", choices=["Noam", "Adam"])
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--clip_grads', action='store_true')

parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--num_frames', type=int, default=30)


def run_trainer(config):
	config['evaluate'] = True
	random.seed(0)
	np.random.seed(0)
	torch.manual_seed(0)

	run_name_format = (
		"d_model={d_model}-"
		"layers_count={layers_count}-"
		"heads_count={heads_count}-"
		"pe={positional_encoding}-"
		"optimizer={optimizer}-"
		"{timestamp}"
	)
	model = build_model(config)
	if config['evaluate']:
		path_model = './checkpoints/frame_triplet_clip_triplet_gps_loss/best_model.pth'
		print(path_model)
		model.load_state_dict(torch.load(path_model))
		print("Pretrained model loaded! ")
	num_frames = config['num_frames']

	import os

	basepath = '.'
	feats_dir = basepath + '/NetVLAD_Feats_SF_train/'
	list0 = next(os.walk(os.path.join(feats_dir, 'train')))[2]
	list0 = [os.path.join(feats_dir, 'train', items) for items in list0]
	train_num_frames = 8
	train_dataset = BDD_GSV_EncoderFeats_Dataset(feats_dir, phase='train', clip_len=num_frames,
																  num_frames=train_num_frames, feat_files=list0,
																  city_name='sf_train')
	train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0, drop_last=True)



	feats_dir = basepath + '/NetVLAD_Feats_SF_val/'
	list1 = next(os.walk(os.path.join(feats_dir, 'val')))[2]
	list1 = [os.path.join(feats_dir, 'val', items) for items in list1]
	val_dataset1 = BDD_GSV_EncoderFeats_Dataset(feats_dir, phase='val', clip_len=num_frames,
																 num_frames=num_frames, feat_files=list1,
																 city_name='sf')
	val_dataloader1 = DataLoader(val_dataset1, batch_size=1, shuffle=False, num_workers=0, drop_last=True)



	feats_dir = basepath + '/NetVLAD_Feats_Berkeley/'
	list2 = next(os.walk(os.path.join(feats_dir, 'val')))[2]
	list2 = [os.path.join(feats_dir, 'val', items) for items in list2]
	val_dataset2 = BDD_GSV_EncoderFeats_Dataset(feats_dir, phase='val', clip_len=num_frames,
																 num_frames=num_frames, feat_files=list2,
																 city_name='berkeley')
	val_dataloader2 = DataLoader(val_dataset2, batch_size=1, shuffle=False, num_workers=0, drop_last=True)



	feats_dir = basepath + '/NetVLAD_Feats_NY/'
	list3 = next(os.walk(os.path.join(feats_dir, 'val')))[2]
	list3 = [os.path.join(feats_dir, 'val', items) for items in list3]
	val_dataset3 = BDD_GSV_EncoderFeats_Dataset(feats_dir, phase='val', clip_len=num_frames,
																 num_frames=num_frames, feat_files=list3,
																 city_name='ny')
	val_dataloader3 = DataLoader(val_dataset3, batch_size=1, shuffle=False, num_workers=0, drop_last=True)




	feats_dir = basepath + '/NetVLAD_Feats_BayArea'
	list4 = next(os.walk(os.path.join(feats_dir, 'val')))[2]
	list4 = [os.path.join(feats_dir, 'train', items) for items in list4]
	val_dataset4 = BDD_GSV_EncoderFeats_Dataset(feats_dir, phase='val', clip_len=num_frames,
																 num_frames=num_frames, feat_files=list4,
																 city_name='bayarea')
	val_dataloader4 = DataLoader(val_dataset4, batch_size=1, shuffle=False, num_workers=0, drop_last=True)



	val_dataloader = [val_dataloader1, val_dataloader2, val_dataloader3, val_dataloader4]

	print(len(train_dataloader))
	for d in val_dataloader:
		print(len(d))


	loss_function = TripletLoss(2.0)

	accuracy_function = AccuracyMetric()

	if config['optimizer'] == 'Noam':
		optimizer = NoamOptimizer(model.parameters(), d_model=config['d_model'])
	elif config['optimizer'] == 'Adam':
		optimizer = Adam(model.parameters(), lr=config['lr'])
	else:
		raise NotImplementedError()

	logger = None
	run_name = None
	trainer = GTFL_Trainer(
		model=model,
		train_dataloader=train_dataloader,
		val_dataloader=val_dataloader,
		loss_function=loss_function,
		metric_function=accuracy_function,
		optimizer=optimizer,
		logger=logger,
		run_name=run_name,
		save_config=config['save_config'],
		save_checkpoint=config['save_checkpoint'],
		config=config
	)

	trainer.run(config['epochs'])

	return trainer


if __name__ == '__main__':

	args = parser.parse_args()

	if args.config is not None:
		with open(args.config) as f:
			config = json.load(f)

		default_config = vars(args)
		for key, default_value in default_config.items():
			if key not in config:
				config[key] = default_value
	else:
		config = vars(args)

	run_trainer(config)
