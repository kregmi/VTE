from models import build_model
# from datasets import IndexedInputTargetTranslationDataset
from data_loader_bdd_gsv_feats import BDD_GSV_Smoothing_Trajectory_DataLoader
# from dictionaries import IndexDictionary
from losses import TokenCrossEntropyLoss, LabelSmoothingLoss, TripletLoss
from metrics import AccuracyMetric
from optimizers import NoamOptimizer
from trainer import EpochSeq2SeqTrainer
from utils.log import get_logger
from utils.pipe import input_target_collate_fn

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
import numpy as np

from argparse import ArgumentParser
from datetime import datetime
import json
import random
import torchvision.transforms as transforms
import  os

import params_bdd_gsv as params



parser = ArgumentParser(description='Train Transformer')
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
parser.add_argument('--use_clip_loss', action='store_false')
parser.add_argument('--use_gps_loss', action='store_false')

parser.add_argument('--d_model', type=int, default=512)
parser.add_argument('--in_feat_size', type=int, default=32768)
parser.add_argument('--layers_count', type=int, default=2)
parser.add_argument('--heads_count', type=int, default=2)
parser.add_argument('--d_ff', type=int, default=128)
parser.add_argument('--dropout_prob', type=float, default=0.2)

parser.add_argument('--label_smoothing', type=float, default=0.1)
parser.add_argument('--optimizer', type=str, default="Adam", choices=["Noam", "Adam"])
parser.add_argument('--lr', type=float, default=0.00001)
parser.add_argument('--clip_grads', action='store_true')

parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--num_frames', type=int, default=30)


def run_trainer(config):
	config['evaluate'] = True

	num_frames = config['num_frames']
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



	run_name = run_name_format.format(**config, timestamp=datetime.now().date().strftime("%Y_%m_%d"))
	logger = get_logger(run_name, save_log=config['save_log'])
	logger.info(config)

	logger.info('Building model...')
	model = build_model(config)
	if config['evaluate']:
		path_model = './checkpoints/best_model.pth'

		model.load_state_dict(torch.load(path_model))
		print("Pretrained model loaded! ")


	logger.info(model)
	logger.info('Total : {parameters_count} parameters'.format(parameters_count=sum([p.nelement() for p in model.parameters()])))

	logger.info('Loading datasets...')



	feats_file0 = './'  # PATH TO TRAIN FEATS
	gps_limit0 = [37.65, 37.81, -122.5, -122.38]
	train_num_frames = 8
	train_dataset = BDD_GSV_Smoothing_Trajectory_DataLoader(feats_file0, phase='train', clip_len=num_frames,
															num_frames=train_num_frames, gps_limit=gps_limit0)
	train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0, drop_last=True)


	feats_file1 = './'  # PATH TO SF FEATS
	feats_file2 = './'  # PATH TO BAY AREA FEATS
	feats_file3 = './'  # PATH TO BERKELEY FEATS
	feats_file4 = './'  # PATH TO NEW YORK FEATS


	gps_limit1 = [37.65, 37.81, -122.5, -122.38]
	gps_limit2 = [37.419279, 37.507089, -122.258048, -122.1054]
	gps_limit3 = [37.72409913, 37.897474, -122.312608, -122.100853]
	gps_limit4 = [40.7073, 40.7381, -74.01486, -73.9687]




	val_dataset1 = BDD_GSV_Smoothing_Trajectory_DataLoader(feats_file1, phase='val', clip_len=num_frames,
														   num_frames=num_frames, gps_limit=gps_limit1)
	val_dataloader1 = DataLoader(val_dataset1, batch_size=1, shuffle=False, num_workers=0, drop_last=True)



	val_dataset2 = BDD_GSV_Smoothing_Trajectory_DataLoader(feats_file2, phase='val', clip_len=num_frames,
														   num_frames=num_frames, gps_limit=gps_limit2)
	val_dataloader2 = DataLoader(val_dataset2, batch_size=1, shuffle=False, num_workers=0, drop_last=True)



	val_dataset3 = BDD_GSV_Smoothing_Trajectory_DataLoader(feats_file3, phase='val', clip_len=num_frames,
														   num_frames=num_frames, gps_limit=gps_limit3)
	val_dataloader3 = DataLoader(val_dataset3, batch_size=1, shuffle=False, num_workers=0, drop_last=True)




	val_dataset4 = BDD_GSV_Smoothing_Trajectory_DataLoader(feats_file4, phase='val', clip_len=num_frames,
														   num_frames=num_frames, gps_limit=gps_limit4)
	val_dataloader4 = DataLoader(val_dataset4, batch_size=1, shuffle=False, num_workers=0, drop_last=True)



	val_dataloader = [val_dataloader1, val_dataloader2, val_dataloader3, val_dataloader4]


	print(len(train_dataloader))


	loss_function  = torch.nn.MSELoss()

	accuracy_function = AccuracyMetric()

	if config['optimizer'] == 'Noam':
		optimizer = NoamOptimizer(model.parameters(), d_model=config['d_model'])
	elif config['optimizer'] == 'Adam':
		optimizer = Adam(model.parameters(), lr=config['lr'])
	else:
		raise NotImplementedError()

	logger.info('Start training...')
	trainer = EpochSeq2SeqTrainer(
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
