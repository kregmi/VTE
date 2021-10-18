import torch
import numpy as np

from os.path import dirname, abspath, join, exists
from os import makedirs
from datetime import datetime
import json
from geopy.distance import lonlat, distance
import scipy.io as sio


PAD_INDEX = 0

BASE_DIR = dirname(abspath(__file__))


class TrajectorySmoothing:

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
		self.compute_delta = True

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


	def denormalize_lat_lon_coordinates(self, x, min_x, max_x):
		for i in range(len(x)):
			x[i] = float(min_x) + x[i] * (float(max_x) - float(min_x))
		return x


	def compute_dist_trajectory(self, gt, pred):
		dist = []
		for i in range(len(gt)):
			dist.append(self.compute_distance_in_meters(gt[i][0], gt[i][1], pred[i][0], pred[i][1]))
		return  dist


	def run_val_epoch(self, dataloader, mode='val'):
		batch_losses = []
		batch_counts = []
		num_frames = self.config['num_frames']
		num_test_imgs = len(dataloader) * num_frames
		batch_size= dataloader.batch_size
		curr_idx = 0
		dist_error = []
		loss_prediction = torch.nn.BCEWithLogitsLoss().cuda()
		noisy_lats = np.zeros((num_test_imgs))
		noisy_lons = np.zeros((num_test_imgs))
		smoothed_lats = np.zeros((num_test_imgs))
		smoothed_lons = np.zeros((num_test_imgs))
		gt_lats = np.zeros((num_test_imgs))
		gt_lons = np.zeros((num_test_imgs))

		with torch.no_grad():
			for batch_idx, (noisy_lat, noisy_lon, gt_lat, gt_lon, noisy_lat_denorm, noisy_lon_denorm, gt_lat_denorm, gt_lon_denorm) in enumerate(dataloader):
				noisy_gps = torch.cat((torch.unsqueeze(noisy_lat, 2), torch.unsqueeze(noisy_lon, 2)), 2)
				gt_gps = torch.cat((torch.unsqueeze(gt_lat, 2), torch.unsqueeze(gt_lon, 2)), 2)
				noisy_gps, gt_gps = (noisy_gps.float()).to(self.device), (gt_gps.float()).to(self.device)
				label_confidence = torch.eq(gt_gps, noisy_gps)
				# target_confidence = torch.where(label_confidence == True, torch.tensor(0.0).cuda(),
				# 								torch.tensor(1.0).cuda())

				if self.compute_delta:
					smoothed_gps_delta, confidence = self.model(noisy_gps)
					pred_confidence = torch.where(torch.sigmoid(confidence) < 0.5, torch.tensor(0.0).cuda(),
												  torch.tensor(1.0).cuda())

					smoothed_gps_delta = torch.where(pred_confidence == 0.0, torch.tensor(0.0).cuda(),
													 smoothed_gps_delta)
					smoothed_gps = noisy_gps + 0.01 * smoothed_gps_delta
				else:
					smoothed_gps, confidences = self.model(noisy_gps)

				batch_loss = self.loss_function(smoothed_gps.view(-1, 2), gt_gps.view(-1, 2))
				batch_count = noisy_gps.shape[0]
				smoothed_lat_denorm = self.denormalize_lat_lon_coordinates(smoothed_gps[:,:,0], dataloader.dataset.gps_limit[0], dataloader.dataset.gps_limit[1])
				smoothed_lon_denorm = self.denormalize_lat_lon_coordinates(smoothed_gps[:,:,1], dataloader.dataset.gps_limit[2], dataloader.dataset.gps_limit[3])
				gt_lat_denorm = self.denormalize_lat_lon_coordinates(gt_lat,
																	 dataloader.dataset.gps_limit[0],
																	 dataloader.dataset.gps_limit[1])
				gt_lon_denorm = self.denormalize_lat_lon_coordinates(gt_lon,
																	 dataloader.dataset.gps_limit[2],
																	 dataloader.dataset.gps_limit[3])

				smoothed_traj = torch.cat((torch.unsqueeze(smoothed_lat_denorm, 2), torch.unsqueeze(smoothed_lon_denorm, 2)), 2)
				gt_traj = torch.cat((torch.unsqueeze(gt_lat_denorm, 2), torch.unsqueeze(gt_lon_denorm, 2)), 2)
				err_dist = self.compute_dist_trajectory(smoothed_traj.squeeze(), gt_traj.squeeze())
				dist_error.append(err_dist)


				batch_losses.append(batch_loss.item())
				batch_counts.append(batch_count)

				noisy_lats[curr_idx:curr_idx + batch_size * num_frames] = (self.denormalize_lat_lon_coordinates(noisy_lat,
																												dataloader.dataset.gps_limit[0],
																												dataloader.dataset.gps_limit[1]).cpu())[0, :]
				noisy_lons[curr_idx:curr_idx + batch_size * num_frames] = (self.denormalize_lat_lon_coordinates(noisy_lon,
																												dataloader.dataset.gps_limit[2],
																												dataloader.dataset.gps_limit[3]).cpu())[0, :]
				smoothed_lats[curr_idx:curr_idx + batch_size * num_frames] = smoothed_lat_denorm[0, :].cpu()
				smoothed_lons[curr_idx:curr_idx + batch_size * num_frames] = smoothed_lon_denorm[0, :].cpu()
				gt_lats[curr_idx:curr_idx + batch_size * num_frames] = gt_lat_denorm[0, :].cpu()
				gt_lons[curr_idx:curr_idx + batch_size * num_frames] = gt_lon_denorm[0, :].cpu()

				curr_idx = curr_idx + batch_size*num_frames


		print('\n   compute accuracy')
		dists_per_traj = ([sum(d)/30 for i, d in enumerate(dist_error)])
		error_meters = sum(dists_per_traj) / (len(dists_per_traj) )
		top1 = 0.0
		length = self.compute_distance_in_meters(dataloader.dataset.gps_limit[0], dataloader.dataset.gps_limit[2], dataloader.dataset.gps_limit[0], dataloader.dataset.gps_limit[3])
		width = self.compute_distance_in_meters(dataloader.dataset.gps_limit[0], dataloader.dataset.gps_limit[2], dataloader.dataset.gps_limit[1], dataloader.dataset.gps_limit[2])
		print ("GPS Window : ")
		print(dataloader.dataset.gps_limit)
		print("Area in square kilometers: ", (length*width)/1000000)
		print("GPS Window of : length = " + str(length) + " width = " + str(width) )
		print("Average localization error: ", error_meters)

		return top1, [error_meters]



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