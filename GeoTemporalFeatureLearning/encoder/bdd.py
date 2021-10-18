from random import randint, random
from utils.video_transforms import  *
from utils.volume_transforms import  *

import pickle
import json
import os

import torch.utils.data as data
from PIL import Image
import  numpy as np


def my_transform():
	video_transform_list = [
		ClipToTensor(channel_nb=3),
		Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
	]
	video_transform = Compose(video_transform_list)
	return  video_transform



class BDD_GSV_Loader(data.Dataset):
	"""docstring for BDD_GSV_Loader
	Loads the data (images and GPS values) for train/val

	"""

	def __init__(self, arg, transform_bdd=None, transform_gsv=None, gsv4imgs=False, flag='val'):
		super(BDD_GSV_Loader, self).__init__()
		self.arg = arg

		self.available_data()
		self.read_pickle()
		self._read_txt_file()
		self.get_daytime_videos()
		self.trajectories_from_given_gps_window()
		self.gsv4imgs = gsv4imgs
		self.flag = flag
		self.transform = my_transform()



	def read_pickle(self):
		with open(self.arg.pickle_file, 'rb') as f:
			self.data = pickle.load(f)
		print("Data size: ", len(self.data))



	def get_daytime_videos(self):
		with open(self.arg.label_json) as f:
			all_videos = json.load(f)
		self.daytime_videos = []
		for i in range (len(all_videos)):
			if all_videos[i]['attributes']['timeofday']  == 'daytime' :
				self.daytime_videos.append(all_videos[i])
		self.daytime_video_names = [vid['name'][:-4] for vid in self.daytime_videos]
		self.data = {key: val for key, val in self.data.items() if key in self.daytime_video_names}
		print("Total daytime videos: ", len(self.data))


	# return daytime_videos
	def _read_txt_file(self):
		bdd_gsv_img_mapper = {}
		with open(self.arg.bdd_gsv_filename_mapper) as file:
			for line in file:
				data1, data2 = line.rstrip('\n').split(',')
				bdd_gsv_img_mapper[data1] = data2
		self.bdd_gsv_img_mapper = bdd_gsv_img_mapper


	def trajectories_from_given_gps_window(self):
		self.trajectories = []
		for k, v in enumerate(self.data):
			if v in self.dirs_downloaded:
				latitudes = []
				longitudes = []
				imgs_bdd = []
				imgs_gsv = []
				if self.arg.gps_limit[2] < self.data[v][0][2] < self.arg.gps_limit[3] and \
										self.arg.gps_limit[0] < self.data[v][0][1] < self.arg.gps_limit[1]:
					if(len(self.data[v])>=30):
						for j in range(len(self.data[v])):
							try:
								latitudes.append(self.data[v][j][1])
								longitudes.append(self.data[v][j][2])
								bdd = self.arg.bdd_dir + '/' + v + '/' + self.data[v][j][0]
								gsv = self.bdd_gsv_img_mapper[bdd]
								imgs_bdd.append(bdd)
								imgs_gsv.append(gsv)

							except:
								pass
						if len(imgs_gsv) >= 30:
							self.trajectories.append([v, [imgs_bdd, imgs_gsv, latitudes, longitudes]])
		print("Number of Trajectories in given gps window: ", len(self.trajectories))


	def available_data(self):
		self.dirs_downloaded = next(os.walk(self.arg.gsv_dir))[1]


	def _normalize_gps(self):
		self.min_lat, self.max_lat = 90, -90
		self.min_lon, self.max_lon = 180, -180

		for i in range(len(self.trajectories)):
			curr_max_lat, curr_min_lat = max(self.trajectories[i][1][2][:]) , min(self.trajectories[i][1][2][:])
			if curr_max_lat > self.max_lat:
				self.max_lat = curr_max_lat
			if curr_min_lat < self.min_lat:
				self.min_lat = curr_min_lat

			curr_max_lon, curr_min_lon = max(self.trajectories[i][1][3][:]) , min(self.trajectories[i][1][3][:])
			if curr_max_lon > self.max_lon:
				self.max_lon = curr_max_lon
			if curr_min_lon < self.min_lon:
				self.min_lon = curr_min_lon

		for i in range(len(self.trajectories)):
			self.trajectories[i][1][2][:] = (self.trajectories[i][1][2][:] - self.min_lat)/(self.max_lat - self.min_lat)
			self.trajectories[i][1][3][:] = (self.trajectories[i][1][3][:] - self.min_lon)/(self.max_lon - self.min_lon)



	def __len__(self):
		return len(self.trajectories)

	def __getitem__(self, index):
		record = self.trajectories[index]
		if self.flag == 'train':
			start_idx = random.randint(0, self.arg.timesteps - self.arg.num_frames - 1)
			segment_indices = start_idx + np.arange(self.arg.num_frames)
		else:
			# start_idx = 0
			segment_indices = np.arange(self.arg.timesteps)
		lat, lon = self.get_gps(record[1], segment_indices)
		dir_name = record[0]

		return self.get_imgs(record[1][0], segment_indices, 'bdd'), self.get_imgs(record[1][1], segment_indices, 'gsv'), dir_name, lat, lon


	def get_imgs(self, record, indices, img_source):
		images = list()
		if img_source == 'bdd':
			for ind in indices:
				seg_imgs = [Image.open(record[ind]).convert('RGB')]
				images.extend(seg_imgs)
		elif img_source == 'gsv':
			for ind in indices:
				if self.gsv4imgs:
					for i in range(4):  # if I want to save features for all 4 images at a gps location
						seg_imgs = [Image.open(record[ind][:-10] + '/gsv_' + str(i) + '.jpg').convert('RGB')]
						images.extend(seg_imgs)
				else:
					seg_imgs = [Image.open(record[ind]).convert('RGB')]
					images.extend(seg_imgs)
		else:
			exit()

		process_data = self.transform(images)
		return process_data

	def get_gps(self, record, indices):
		lat = record[2][indices[0]:indices[0] + len(indices)]
		lon = record[3][indices[0]:indices[0] + len(indices)]
		return  lat, lon






class BDD_GSV_EVAL_Loader(data.Dataset):
	"""docstring for BDD_GSV_EVAL_Loader
	Loads the data (images and GPS values) for train/val

	"""
	def __init__(self, arg, transform_bdd=None, transform_gsv=None,):
		super(BDD_GSV_EVAL_Loader, self).__init__()
		self.arg = arg

		self.read_pickle()
		self._read_txt_file()
		self.get_daytime_videos()
		self.trajectories_from_given_gps_window()
		self.available_data_pairs()
		self.transform = my_transform()



	def read_pickle(self):
		with open(self.arg.pickle_file, 'rb') as f:
			self.data = pickle.load(f)
		print("Data size: ", len(self.data))



	def get_daytime_videos(self):
		with open(self.arg.label_json) as f:
			all_videos = json.load(f)
		self.daytime_videos = []
		for i in range (len(all_videos)):
			if all_videos[i]['attributes']['timeofday']  == 'daytime' :
				self.daytime_videos.append(all_videos[i])
		self.daytime_video_names = [vid['name'][:-4] for vid in self.daytime_videos]
		self.data = {key: val for key, val in self.data.items() if key in self.daytime_video_names}
		print("Total daytime videos: ", len(self.data))


	# return daytime_videos
	def _read_txt_file(self):
		bdd_gsv_img_mapper = {}
		with open(self.arg.bdd_gsv_filename_mapper) as file:
			for line in file:
				data1, data2 = line.rstrip('\n').split(',')
				bdd_gsv_img_mapper[data1] = data2
		self.bdd_gsv_img_mapper = bdd_gsv_img_mapper


	def trajectories_from_given_gps_window(self):
		self.selected_data = []
		for k, v in enumerate(self.data):
			latitudes = []
			longitudes = []
			imgs_bdd = []
			imgs_gsv = []
			if self.arg.gps_limit[2] < self.data[v][0][2] < self.arg.gps_limit[3] and \
									self.arg.gps_limit[0] < self.data[v][0][1] < self.arg.gps_limit[1]:
				if(len(self.data[v])>30):
					for j in range(len(self.data[v])):
						try:
							latitudes.append(self.data[v][j][1])
							longitudes.append(self.data[v][j][2])
							bdd = self.arg.bdd_dir + '/' + v + '/' + self.data[v][j][0]
							gsv = self.arg.gsv_dir + '/' + v + '/' + self.data[v][j][0]
							imgs_bdd.append(bdd)
							imgs_gsv.append(gsv)
						except:
							pass
					if len(imgs_gsv) > 30:
						self.selected_data.append([v, [imgs_bdd, imgs_gsv, latitudes, longitudes]])
		print("Number of Trajectories in given gps window: ", len(self.selected_data))


	def available_data_pairs(self):
		self.dirs_downloaded = next(os.walk(self.arg.gsv_dir))[1]
		dirs_feats =  next(os.walk('../Dataset/VTE_BDD_GSV_Dataset/NetVLAD_Feats/val'))[2]
		self.dirs_feats = [d[:-4] for d in dirs_feats]
		self.trajectories = [self.selected_data[i] for i in range(len(self.selected_data)) if
							 self.selected_data[i][0] in self.dirs_downloaded]


	def _normalize_gps(self):
		self.min_lat, self.max_lat = 90, -90
		self.min_lon, self.max_lon = 180, -180

		for i in range(len(self.trajectories)):
			curr_max_lat, curr_min_lat = max(self.trajectories[i][1][2][:]) , min(self.trajectories[i][1][2][:])
			if curr_max_lat > self.max_lat:
				self.max_lat = curr_max_lat
			if curr_min_lat < self.min_lat:
				self.min_lat = curr_min_lat

			curr_max_lon, curr_min_lon = max(self.trajectories[i][1][3][:]) , min(self.trajectories[i][1][3][:])
			if curr_max_lon > self.max_lon:
				self.max_lon = curr_max_lon
			if curr_min_lon < self.min_lon:
				self.min_lon = curr_min_lon

		for i in range(len(self.trajectories)):
			self.trajectories[i][1][2][:] = (self.trajectories[i][1][2][:] - self.min_lat)/(self.max_lat - self.min_lat)
			self.trajectories[i][1][3][:] = (self.trajectories[i][1][3][:] - self.min_lon)/(self.max_lon - self.min_lon)


	def __len__(self):
		return len(self.trajectories)

	def __getitem__(self, index):
		index = index + 61
		record = self.trajectories[index]
		segment_indices = np.arange(self.arg.timesteps)
		lat, lon = self.get_gps(record[1], segment_indices)
		dir_name = record[0]

		return self.get_imgs(record[1][0], segment_indices, 'bdd'), self.get_imgs(record[1][1], segment_indices, 'gsv'), dir_name, lat, lon


	def get_imgs(self, record, indices, img_source):
		images = list()
		if img_source == 'bdd':
			for ind in indices:
				seg_imgs = [Image.open(record[ind]).convert('RGB')]
				images.extend(seg_imgs)

		elif img_source == 'gsv':
			for ind in indices:
				seg_imgs = [Image.open(record[ind]).convert('RGB')]
				images.extend(seg_imgs)
		else:
			exit()

		process_data = self.transform(images)

		return process_data

	def get_gps(self, record, indices):
		lat = record[2][indices[0]:indices[0] + self.arg.timesteps]
		lon = record[3][indices[0]:indices[0] + self.arg.timesteps]
		return  lat, lon


