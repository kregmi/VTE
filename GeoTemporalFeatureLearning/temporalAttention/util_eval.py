# import torch

# from torch import nn
import numpy as np
from geopy.distance import lonlat, distance

def accuracy_topN(dist_array, topN):
    accuracy = 0.0
    data_amount = 0.0
    for i in range(dist_array.shape[0]):
        gt_dist = dist_array[i, i]
        prediction = np.sum(dist_array[:, i] < gt_dist)
        if prediction < topN:
            accuracy += 1.0
        data_amount += 1
    accuracy /= data_amount
    accuracy *= 100.0
    return accuracy

def compute_accuracy(dist_array):
    topN = 10
    top10_accuracy = accuracy_topN(dist_array, topN)
    print('Accuracy for top ' + str(topN) + ': ', top10_accuracy)
    topN = 1
    top1_accuracy = accuracy_topN(dist_array, topN)
    print('Accuracy for top ' + str(topN) + ': ', top1_accuracy)

    return top10_accuracy, top1_accuracy


def accuracy_topN_4gsv(dist_array, topN):
    accuracy = 0.0
    data_amount = 0.0
    for i in range(dist_array.shape[0]):
        gt_dist = dist_array[i, i]
        prediction = np.sum(dist_array[:, i] < gt_dist)
        if prediction < topN:
            accuracy += 1.0
        data_amount += 1
    accuracy /= data_amount
    accuracy *= 100.0
    return accuracy


def compute_accuracy_4gsv(dist_array):
    topN = 10
    top10_accuracy = accuracy_topN(dist_array, topN)
    print('Accuracy for top ' + str(topN) + ': ', top10_accuracy)
    topN = 1
    top1_accuracy = accuracy_topN(dist_array, topN)
    print('Accuracy for top ' + str(topN) + ': ', top1_accuracy)

    return top10_accuracy, top1_accuracy

def denormalize_lat_lon_coordinates(x, minmax_x_y):
    for j in range(x.shape[0]):
        for i in range(x.shape[1]):
            x[j][i][0] = float(minmax_x_y[0]) + x[j][i][0] * (float(minmax_x_y[1]) - float(minmax_x_y[0]))
            x[j][i][1] = float(minmax_x_y[2]) + x[j][i][1] * (float(minmax_x_y[3]) - float(minmax_x_y[2]))
    return x

def compute_error_distance(ref_db_lat, ref_db_lon, val_lat, val_lon, idx_sorted):
    dist = 0.0
    distarray = []
    for i in range(idx_sorted.shape[0]):
        # dist += compute_distance_in_meters(val_lat[i], val_lon[i], ref_db_lat[idx_sorted[i][0]], ref_db_lon[idx_sorted[i][0]])
        distarray.append(compute_distance_in_meters(val_lat[i], val_lon[i], ref_db_lat[idx_sorted[i][0]],
                                       ref_db_lon[idx_sorted[i][0]]))
    # counts = sorted(distarray)
    # plot_auc(distarray, args.data_split_type)
    # sio.savemat(os.path.join('./checkpoints', args.checkpoint_name, 'distarray' + str(args.epoch) + '_' + str(args.data_split_type) + '.mat'), {'distarray': distarray})

    print("Traj 1 Error: ", sum(distarray[:30])/30)
    print("Traj 5 Error: ", sum(distarray[120:150]) / 30)
    dists = [ sum(distarray[i*30:(i+1)*30])/30 for i in range(len(distarray)//30)]
    for i in range(len(distarray) // 30):
        print(distarray[i*30:(i+1)*30]  )
    # print(dists)
    return dists


def compute_distance_in_meters(x1, y1, x2, y2):
    return distance(lonlat(*(y1, x1)), lonlat(*(y2, x2))).m


def error_in_meters(sorted_idx, lat, lon):
	return compute_error_distance(lat, lon, lat, lon, sorted_idx)


def compute_error_distance(ref_db_lat, ref_db_lon, val_lat, val_lon, idx_sorted):
	distarray = []
	for i in range(idx_sorted.shape[0]):
		distarray.append(compute_distance_in_meters(val_lat[i], val_lon[i], ref_db_lat[idx_sorted[i][0]],
														 ref_db_lon[idx_sorted[i][0]]))
	return distarray
