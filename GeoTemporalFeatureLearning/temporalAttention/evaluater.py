import torch
import numpy as np

from os.path import dirname, abspath
from datetime import datetime
from geopy.distance import lonlat, distance
import scipy.io as sio


PAD_INDEX = 0

BASE_DIR = dirname(abspath(__file__))


class GTFL_Trainer:

	def __init__(self, model,
				 train_dataloader, val_dataloader,
				 loss_function, metric_function, optimizer,
				 logger, run_name,
				 save_config, save_checkpoint,
				 config):

		self.config = config
		self.device = torch.device(self.config['device'])

		self.model = model.to(self.device)
		self.train_dataloader = train_dataloader
		self.val_dataloader = val_dataloader

		self.loss_function = loss_function.to(self.device)
		self.metric_function = metric_function
		self.optimizer = optimizer
		self.clip_grads = self.config['clip_grads']

		self.print_every = self.config['print_every']
		self.save_every = self.config['save_every']

		self.epoch = 0
		self.history = []

		self.start_time = datetime.now()

		self.best_val_metric = None
		self.best_checkpoint_filepath = None

		self.save_checkpoint = save_checkpoint
		self.save_format = 'epoch={epoch:0>3}-val_loss={val_loss:<.3}-val_metrics={val_metrics}.pth'

		self.log_format = (
			"Epoch: {epoch:>3} "
			"Progress: {progress:<.1%} "
			"Elapsed: {elapsed} "
			"Examples/second: {per_second:<.1} "
			"Train Loss: {train_loss:<.6} "
			"Val Loss: {val_loss:<.6} "
			"Train Metrics: {train_metrics} "
			"Val Metrics: {val_metrics} "
			"Learning rate: {current_lr:<.4} "
		)

	def compute_distance_in_meters(self, x1, y1, x2, y2):
		return distance(lonlat(*(y1, x1)), lonlat(*(y2, x2))).m

	def validate(self, dist_array, top_k):
		accuracy = 0.0
		data_amount = 0.0
		for i in range(dist_array.shape[0]):
			gt_dist = dist_array[i, i]
			prediction = np.sum(dist_array[:, i] < gt_dist)
			if prediction < top_k:
				accuracy += 1.0
			data_amount += 1.0
		accuracy /= data_amount

		return accuracy * 100

	def compute_error_distance(self, ref_db_lat, ref_db_lon, val_lat, val_lon, idx_sorted):
		dist = 0.0
		distarray = []
		for i in range(idx_sorted.shape[0]):
			distarray.append(self.compute_distance_in_meters(val_lat[i], val_lon[i], ref_db_lat[idx_sorted[i][0]],
														ref_db_lon[idx_sorted[i][0]]))
		return distarray

	def error_in_meters(self, sorted_idx, lat, lon):
		return self.compute_error_distance(lat, lon, lat, lon, sorted_idx)


	def run_val_epoch(self, dataloader, mode='val'):
		num_frames = self.config['num_frames']
		num_test_imgs = len(dataloader) * num_frames
		gsv_descriptors = np.zeros((num_test_imgs, self.config['d_model']))
		bdd_descriptors = np.zeros((num_test_imgs, self.config['d_model']))
		lats = np.zeros((num_test_imgs))
		lons = np.zeros((num_test_imgs))
		batch_size= dataloader.batch_size
		curr_idx = 0

		with torch.no_grad():

			for batch_idx, (clip_bdd, clip_gsv, lat, lon) in enumerate(dataloader):
				clip_bdd, clip_gsv = (clip_bdd.float()).to(self.device), (clip_gsv.float()).to(self.device)
				output_bdd, op_bdd = self.model(clip_bdd, mask=None)
				output_gsv, op_gsv = self.model(clip_gsv.view(-1,1,clip_gsv.shape[2]), mask=None)
				output_gsv = output_gsv.view(1, -1, output_gsv.shape[2])

				bdd_descriptors[curr_idx:curr_idx + batch_size*num_frames, :] = (output_bdd[0,:,:].to("cpu", torch.double))
				gsv_descriptors[curr_idx:curr_idx + batch_size*num_frames, :] = (output_gsv[0,:,:].to("cpu", torch.double))
				lats[curr_idx:curr_idx + batch_size * num_frames] = lat[0,:]
				lons[curr_idx:curr_idx + batch_size * num_frames] = lon[0,:]
				curr_idx = curr_idx + batch_size*num_frames


		print('   compute accuracy')
		dist_array = 2 - 2 * np.matmul(bdd_descriptors, np.transpose(gsv_descriptors))
		idx_sorted = np.argsort(dist_array, axis=1)
		distarray = self.error_in_meters(idx_sorted, lats, lons)
		error_meters = sum(distarray) / len(distarray)

		top1_percent = int(dist_array.shape[0] * 0.01) + 1
		val_accuracy = np.zeros((1, top1_percent))
		for i in range(top1_percent):
			val_accuracy[0, i] = self.validate(dist_array, i)

		top1 = val_accuracy[0, 1]
		top5 = val_accuracy[0, 5]
		print('top1', ':', top1)
		print('top5', ':', top5)

		print("Average localization error: ", error_meters)

		return top5, [error_meters]


	def run(self, epochs=10):
		self.model.eval()
		error_meters_city = []
		for dataloader in self.val_dataloader:
			val_epoch_loss, error_meters = self.run_val_epoch(dataloader, mode='val')
			error_meters_city.append(error_meters[0])
		return


	def _elapsed_time(self):
		now = datetime.now()
		elapsed = now - self.start_time
		return str(elapsed).split('.')[0]