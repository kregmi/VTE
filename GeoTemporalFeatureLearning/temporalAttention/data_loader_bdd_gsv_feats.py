from torch.utils.data import Dataset
import scipy.io as sio
import numpy as np
import  random



class BDD_GSV_EncoderFeats_Dataset(Dataset):
	def __init__(self,
				 feats_dir,
				 phase,
				 clip_len,
				 num_frames,
				 feat_files,
				 city_name):

		super(BDD_GSV_EncoderFeats_Dataset, self).__init__()
		self.feats_dir = feats_dir
		self.phase = phase
		self.num_frames = num_frames
		self.clip_len = clip_len
		self.feat_files = feat_files
		self.city_name = city_name

	def __len__(self):
		return  len(self.feat_files)


	def _get_input_features(self, traj):
		matfile = sio.loadmat((traj))
		if self.clip_len != self.num_frames:
			start_idx = random.randint(0, self.clip_len - self.num_frames - 1)
			bdd = matfile['bdd_feats'][start_idx:self.num_frames + start_idx,:]
			gsv = matfile['gsv_feats'][start_idx:self.num_frames + start_idx,:]
			lat = matfile['lat'][0][start_idx:self.num_frames + start_idx]
			lon = matfile['lon'][0][start_idx:self.num_frames + start_idx]
		else:
			bdd = matfile['bdd_feats'][:self.num_frames, :]
			gsv = matfile['gsv_feats'][:self.num_frames, :]
			lat = matfile['lat'][0][:self.num_frames]
			lon = matfile['lon'][0][:self.num_frames]
		# else:
		# 	exit()
		while bdd.ndim>2:
			bdd = np.squeeze(bdd, axis=bdd.ndim-1)
		while gsv.ndim > 2:
			gsv = np.squeeze(gsv, axis=gsv.ndim - 1)

		return bdd, gsv, lat, lon

	def __getitem__(self, idx):
		bdd, gsv, lat, lon = self._get_input_features(self.feat_files[idx])
		return bdd, gsv, lat, lon
