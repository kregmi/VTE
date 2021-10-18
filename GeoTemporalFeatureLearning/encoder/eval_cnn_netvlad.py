import os
import numpy as np
import scipy.io as sio
from geopy.distance import lonlat, distance


def compute_distance_in_meters( x1, y1, x2, y2):
	return distance(lonlat(*(y1, x1)), lonlat(*(y2, x2))).m


def compute_error_distance(ref_db_lat, ref_db_lon, val_lat, val_lon, idx_sorted):
	distarray = []
	for i in range(idx_sorted.shape[0]):
		distarray.append(compute_distance_in_meters(val_lat[i], val_lon[i], ref_db_lat[idx_sorted[i][0]], ref_db_lon[idx_sorted[i][0]]))
	return distarray

def error_in_meters(sorted_idx, lat, lon):
	return compute_error_distance(lat, lon, lat, lon, sorted_idx)


basepath = '.'
feat_dirs = [basepath + '/NetVLAD_Feats_SF/val',
			 basepath + '/NetVLAD_Feats_NY/val',
			 basepath + '/NetVLAD_Feats_Berkeley/val',
			 basepath + '/NetVLAD_Feats_BayArea/val']

for feat_dir in feat_dirs:
	city_name = feat_dir.split('/')[-2].split('_')[-1]
	print(feat_dir, city_name)
	dirs_feats = next(os.walk(feat_dir))[2]

	feat_files = [ os.path.join(feat_dir, fn ) for fn in dirs_feats]
	print("# Trajectories: ", len(feat_files))

	num_frames = 30
	feat_dim = 32768

	num_views = 1
	bdd_feats = np.zeros((len(feat_files)*num_frames, feat_dim))
	gsv_feats = np.zeros((len(feat_files)*num_frames*num_views, feat_dim))
	similarities = np.zeros((4))
	lats= np.zeros((len(feat_files)*num_frames))
	lons = np.zeros((len(feat_files)*num_frames))

	for i in range(len(feat_files)):
		feats = sio.loadmat(feat_files[i])
		bdd = feats['bdd_feats']
		gsv = feats['gsv_feats']
		bdd_feats[i * num_frames:(i + 1) * num_frames] = bdd[:num_frames]
		gsv_feats[i * num_frames * num_views:(i + 1) * num_frames * num_views] = gsv[:num_frames]
		lats[i * num_frames:(i + 1) * num_frames] = feats['lat'][0][:num_frames]
		lons[i * num_frames:(i + 1) * num_frames] = feats['lon'][0][:num_frames]

	dist_array = 2 - 2 * np.matmul(bdd_feats, np.transpose(gsv_feats))
	idx_sorted = np.argsort(dist_array, axis=1)
	distarray = error_in_meters(idx_sorted, lats, lons)
	error_meters = sum(distarray)/ len(dist_array)
	print(error_meters)


	filename = './eval_feats/eval_result_' + str(city_name) + '_' + str(error_meters)  + '.mat'
	print(filename)

	print("Done!!!")

