from torch.utils.data import Dataset
import scipy.io as sio
import numpy as np
import  random


class BDD_GSV_Smoothing_Trajectory_DataLoader(Dataset):
	def __init__(self,
				 feat_file,
				 phase,
				 clip_len,
				 num_frames,
				 gps_limit):

		super(BDD_GSV_Smoothing_Trajectory_DataLoader, self).__init__()
		self.phase = phase
		self.num_frames = num_frames
		self.clip_len = clip_len
		self.feat_file = feat_file
		self.gps_limit = gps_limit
		self.stddev = 0.0001

		self._load_matfile()
		self.num_clips = len(self.noisy_lats)//self.clip_len
		self._normalize_gps_values()
		self._random_perturbed_inputs()


		print(len(self.noisy_lats))

	def _random_perturbed_inputs(self):
		noise = np.random.randint(10, size=len(self.gt_lons_normalized))
		noise = np.where(noise > 3, 0, noise * self.stddev)
		self.perturbed_lat = self.gt_lats_normalized + noise
		self.perturbed_lon = self.gt_lons_normalized + noise


	def _normalize_gps_values(self):
		self.gt_lats_normalized = self.normalize_lat_lon_coordinates(self.gt_lats, self.gps_limit[0], self.gps_limit[1])
		self.noisy_lats_normalized = self.normalize_lat_lon_coordinates(self.noisy_lats, self.gps_limit[0], self.gps_limit[1])

		self.gt_lons_normalized = self.normalize_lat_lon_coordinates(self.gt_lons, self.gps_limit[2], self.gps_limit[3])
		self.noisy_lons_normalized = self.normalize_lat_lon_coordinates(self.noisy_lons, self.gps_limit[2], self.gps_limit[3])

	def denormalize_lat_lon_coordinates(self, x, min_x, max_x):
		for i in range(len(x)):
			x[i] = float(min_x) + x[i] * (float(max_x) - float(min_x))
		return x

	def normalize_lat_lon_coordinates(self, x, min_x, max_x):
		for i in range(len(x)):
			x[i] = (x[i] - float(min_x))/ (float(max_x) - float(min_x))
		return x

	def _get_noisy_gps(self):
		noisy_lats, noisy_lons = np.zeros((len(self.gt_lats))), np.zeros((len(self.gt_lats)))
		for i in range(len(self.gt_lats)):
			noisy_lats[i] = self.gt_lats[self.sorted_index[i][0]]
			noisy_lons[i] = self.gt_lons[self.sorted_index[i][0]]
		return noisy_lats, noisy_lons

	def _load_matfile(self):
		data = sio.loadmat(self.feat_file)
		try:
			self.gt_lats = data['lat'][0]
			self.gt_lons = data['lon'][0]
			self.sorted_index = data['sorted_array']
			self.gt_lats_denorm = data['lat'][0]
			self.gt_lons_denorm = data['lon'][0]
		except:
			self.gt_lats = data['lats'][0]
			self.gt_lons = data['lons'][0]
			self.sorted_index = data['sorted_idx']
			self.gt_lats_denorm = data['lats'][0]
			self.gt_lons_denorm = data['lons'][0]


		noisy_lats, noisy_lons = self._get_noisy_gps()
		self.noisy_lats, self.noisy_lons = noisy_lats, noisy_lons
		self.noisy_lats_denorm, self.noisy_lons_denorm  = noisy_lats, noisy_lons

		print(len(self.noisy_lats))


	def __len__(self):
		return  self.num_clips

	def _get_noisy_gt_gps(self, idx):
		gt_lat = self.gt_lats_normalized[idx*self.clip_len:(idx+1)*self.clip_len]
		gt_lon = self.gt_lons_normalized[idx*self.clip_len:(idx+1)*self.clip_len]
		if self.phase == 'train':
			if random.random()>0.5:
				noisy_lat = self.perturbed_lat[idx * self.clip_len:(idx + 1) * self.clip_len]
				noisy_lon = self.perturbed_lon[idx * self.clip_len:(idx + 1) * self.clip_len]
			else:
				noisy_lat = self.noisy_lats_normalized[idx * self.clip_len:(idx + 1) * self.clip_len]
				noisy_lon = self.noisy_lons_normalized[idx * self.clip_len:(idx + 1) * self.clip_len]
		else:
			noisy_lat = self.noisy_lats_normalized[idx*self.clip_len:(idx+1)*self.clip_len]
			noisy_lon= self.noisy_lons_normalized[idx*self.clip_len:(idx+1)*self.clip_len]


		return noisy_lat, noisy_lon, gt_lat, gt_lon

	def _get_noisy_gt_gps_denormalized(self, idx):
		gt_lat = self.gt_lats_denorm[idx*self.clip_len:(idx+1)*self.clip_len]
		gt_lon = self.gt_lons_denorm[idx*self.clip_len:(idx+1)*self.clip_len]
		noisy_lat = self.noisy_lats_denorm[idx*self.clip_len:(idx+1)*self.clip_len]
		noisy_lon= self.noisy_lons_denorm[idx*self.clip_len:(idx+1)*self.clip_len]
		return noisy_lat, noisy_lon, gt_lat, gt_lon


	def __getitem__(self, idx):
		if self.phase == 'train':
			noisy_lat, noisy_lon, gt_lat, gt_lon = self._get_noisy_gt_gps(idx)
			return noisy_lat, noisy_lon, gt_lat, gt_lon
		elif self.phase == 'val':
			noisy_lat, noisy_lon, gt_lat, gt_lon = self._get_noisy_gt_gps(idx)
			noisy_lat_denorm, noisy_lon_denorm, gt_lat_denorm, gt_lon_denorm = self._get_noisy_gt_gps_denormalized(idx)
			return noisy_lat, noisy_lon, gt_lat, gt_lon, noisy_lat_denorm, noisy_lon_denorm, gt_lat_denorm, gt_lon_denorm












