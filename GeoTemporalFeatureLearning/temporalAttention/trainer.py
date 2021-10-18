import torch
import numpy as np
from tqdm import tqdm

from os.path import dirname, abspath, join, exists
from os import makedirs
from datetime import datetime
import json
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
		self.min_error = 1000.0
		self.l1_loss = torch.nn.L1Loss()
		self.bce_loss = torch.nn.BCEWithLogitsLoss()
		self.use_gps_loss = config['use_gps_loss']
		self.use_clip_loss = config['use_clip_loss']
		self.loss_function = loss_function.to(self.device)
		self.metric_function = metric_function
		self.optimizer = optimizer
		self.clip_grads = self.config['clip_grads']

		self.logger = logger
		self.checkpoint_dir = join(BASE_DIR, 'checkpoints', run_name)

		if not exists(self.checkpoint_dir):
			makedirs(self.checkpoint_dir)

		if save_config is None:
			config_filepath = join(self.checkpoint_dir, 'config.json')
		else:
			config_filepath = save_config
		with open(config_filepath, 'w') as config_file:
			json.dump(self.config, config_file)

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

		distarray = []
		for i in range(idx_sorted.shape[0]):

			distarray.append(self.compute_distance_in_meters(val_lat[i], val_lon[i], ref_db_lat[idx_sorted[i][0]],
														ref_db_lon[idx_sorted[i][0]]))

		return distarray

	def error_in_meters(self, sorted_idx, lat, lon):
		return self.compute_error_distance(lat, lon, lat, lon, sorted_idx)



	def gps_loss_function(self, bdd_feats, gsv_feats, lat, lon):
		dists_feat_bdd = 2 - 2 * np.dot(bdd_feats.squeeze().cpu().detach().numpy(), np.transpose(bdd_feats.squeeze().cpu().detach().numpy()))
		dists_feat_gsv = 2 - 2 * np.dot(gsv_feats.squeeze().cpu().detach().numpy(), np.transpose(gsv_feats.squeeze().cpu().detach().numpy()))
		dists_gps = np.zeros_like(dists_feat_bdd)
		for i in range(dists_gps.shape[0]):
			for j in range(dists_gps.shape[1]):
				dists_gps
				[i][j] = self.compute_distance_in_meters(lat[i][0], lon[i][0], lat[j][0], lon[j][0])
		dists_gps = 4 * dists_gps/dists_gps.max()
		loss = self.l1_loss(torch.tensor(dists_feat_bdd), torch.tensor(dists_gps)) + self.l1_loss(torch.tensor(dists_feat_gsv), torch.tensor(dists_gps))

		return loss

	def run_val_epoch(self, dataloader, mode='val'):
		batch_losses = []
		batch_counts = []


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
				output_bdd, output_bdd_avg = self.model(clip_bdd, mask=None)
				output_gsv, op_gsv_avg = self.model(clip_gsv.view(-1, 1, clip_gsv.shape[2]), mask=None)
				output_gsv = output_gsv.view(1, -1, output_gsv.shape[2])
				op_gsv_avg = op_gsv_avg.view(1, -1, op_gsv_avg.shape[2])

				batch_loss = self.loss_function(output_bdd, output_gsv)
				batch_count = clip_bdd.shape[0]

				batch_losses.append(batch_loss.item())
				batch_counts.append(batch_count)

				bdd_descriptors[curr_idx:curr_idx + batch_size*num_frames, :] = (output_bdd[0,:,:].to("cpu", torch.double))
				gsv_descriptors[curr_idx:curr_idx + batch_size*num_frames, :] = (output_gsv[0,:,:].to("cpu", torch.double))
				lats[curr_idx:curr_idx + batch_size * num_frames] = lat[0,:]
				lons[curr_idx:curr_idx + batch_size * num_frames] = lon[0,:]
				curr_idx = curr_idx + batch_size*num_frames


		print('\n   compute accuracy')
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
		print("Accuracies : Top 1 {}, Top 5 {}".format(top1, top5))

		print("Average localization error: ", error_meters)

		return top1, [error_meters]




	def run_epoch(self, dataloader, mode='train'):
		batch_losses = []
		batch_counts = []
		for clip_bdd, clip_gsv, lat, lon in tqdm(dataloader):
			clip_bdd, clip_gsv = (clip_bdd.float()).to(self.device), (clip_gsv.float()).to(self.device)
			output_bdd, op_bdd_avg = self.model(clip_bdd, mask=None)
			output_gsv, op_gsv_avg = self.model(clip_gsv.view(-1, 1, clip_gsv.shape[2]), mask=None)
			output_gsv = output_gsv.view(output_bdd.shape[0], output_bdd.shape[1], output_gsv.shape[2])
			op_gsv_avg = op_gsv_avg.view(output_bdd.shape[0], output_bdd.shape[1], op_gsv_avg.shape[2])

			batch_loss = self.loss_function(output_bdd, output_gsv)
			loss =  batch_loss
			if self.use_clip_loss:
				clip_loss = self.loss_function(op_bdd_avg, op_gsv_avg)
				loss = loss +  10*clip_loss
			if self.use_gps_loss:
				gps_loss = self.gps_loss_function(op_bdd_avg, op_gsv_avg, lat, lon)
				loss = loss + 10*gps_loss
			batch_count = clip_bdd.shape[0]

			if mode == 'train':
				self.optimizer.zero_grad()
				loss.backward()
				if self.clip_grads:
					torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
				self.optimizer.step()

			batch_losses.append(batch_loss.item())
			batch_counts.append(batch_count)


		epoch_loss = sum(batch_losses) / sum(batch_counts)
		epoch_perplexity = float(np.exp(epoch_loss))
		epoch_metrics = [epoch_perplexity]

		return epoch_loss, epoch_metrics

	def run(self, epochs=10):
		self.config['evaluate'] =False
		if self.config['evaluate']:
			self.model.eval()
			val_epoch_loss, error_meters = self.run_val_epoch(self.val_dataloader, mode='val')
			print(error_meters)
			return

		else:

			lowest_error = 2000.0
			max_acc = 0.0

			for epoch in range(self.epoch, epochs + 1):
				self.epoch = epoch

				self.model.train()

				epoch_start_time = datetime.now()
				train_epoch_loss, train_epoch_metrics = self.run_epoch(self.train_dataloader, mode='train')
				epoch_end_time = datetime.now()

				self.model.eval()

				error_meters_city = []
				for dataloader in self.val_dataloader:
					val_epoch_loss, error_meters = self.run_val_epoch(dataloader, mode='val')
					error_meters_city.append(error_meters[0])
				top1 = val_epoch_loss

				if epoch % self.print_every == 0 and self.logger:
					per_second = len(self.train_dataloader.dataset) / ((epoch_end_time - epoch_start_time).seconds + 1)
					current_lr = self.optimizer.param_groups[0]['lr']
					log_message = self.log_format.format(epoch=epoch,
														 progress=epoch / epochs,
														 per_second=per_second,
														 train_loss=train_epoch_loss,
														 val_loss =top1,
														 train_metrics=[round(metric, 4) for metric in train_epoch_metrics],
														 val_metrics=[round(metric, 4) for metric in error_meters],
														 current_lr=current_lr,
														 elapsed=self._elapsed_time()
														 )

					self.logger.info(log_message)

				curr_error = sum(error_meters_city)/len(error_meters_city)
				error_meters.append((curr_error))
				print("Current localization error (average) : ", curr_error)
				if curr_error < self.min_error:
					self.min_error = curr_error
					self._save_model(epoch, train_epoch_loss, val_epoch_loss, train_epoch_metrics, error_meters)
					eph = epoch

					print("Minimum localization error : " + str(self.min_error) + " for epoch : " + str(eph))

	def _save_model(self, epoch, train_epoch_loss, val_epoch_loss, train_epoch_metrics, val_epoch_metrics):

		checkpoint_filename = self.save_format.format(
			epoch=epoch,
			val_loss=val_epoch_loss,
			val_metrics = '-'.join(['{:<.3}'.format(v) for v in val_epoch_metrics])
		)

		if self.save_checkpoint is None:
			checkpoint_filepath = join(self.checkpoint_dir, checkpoint_filename)
		else:
			checkpoint_filepath = self.save_checkpoint

		save_state = {
			'epoch': epoch,
			'train_loss': train_epoch_loss,
			'train_metrics': train_epoch_metrics,
			'val_loss': val_epoch_loss,
			'val_metrics': val_epoch_metrics,
			'checkpoint': checkpoint_filepath,
		}

		torch.save(self.model.state_dict(), checkpoint_filepath)
		if self.epoch > 0:
			self.history.append(save_state)

		representative_val_metric = val_epoch_metrics[0]
		if self.best_val_metric is None or self.best_val_metric > representative_val_metric:
			self.best_val_metric = representative_val_metric
			self.val_loss_at_best = val_epoch_loss
			self.train_loss_at_best = train_epoch_loss
			self.train_metrics_at_best = train_epoch_metrics
			self.val_metrics_at_best = val_epoch_metrics
			self.best_checkpoint_filepath = checkpoint_filepath

		if self.logger:
			self.logger.info("Saved model to {}".format(checkpoint_filepath))
			self.logger.info("Current best model is {}".format(self.best_checkpoint_filepath))
			print("\n")

	def _elapsed_time(self):
		now = datetime.now()
		elapsed = now - self.start_time
		return str(elapsed).split('.')[0]