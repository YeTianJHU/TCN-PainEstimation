import numpy as np 
import os
from os.path import isfile, join
import csv

data_dir = '/media/ye/youtube-8/cbvrp_icip_data_video_feature_npy/videos_npy'
label_dir = '/home/ye/Works/CBVRP-ICIP2017/metadata'

def video_number(data_dir):
	"""Get the number of video files in the given direction.
	Args:
		data_dir: The path to all the folders.
	Returns:
		A list of number of video files in each folder."""
	fold_dir = data_dir
	fold_names = [f for f in os.listdir(fold_dir)]
	sort_name = np.sort(fold_names)
	num_video = []
	for id, dir in enumerate(sort_name[1:]):
		file_dir = join(fold_dir,dir)
		num_video.append(len([name for name in os.listdir(file_dir) if isfile(join(file_dir, name))]))
	return num_video


def load_y(label_dir, file_name, num_video):
	"""Load two kinds of label data.
	Args:
		dir_name: The path to the folder csv files.
		file_name: Name of the csv file, should be neither 'ground_truth.csv' or 'ground_truth_v.csv'.
		num_video: A list of number of video files in each folder.
	Returns:
		A numpy array file of all the labels. The dimention should be sum(num_video[:40]) x num_classes."""
	total_videos = sum(num_video)
	if file_name == 'ground_truth.csv':
		with open(join(label_dir,file_name)) as csvfile:
			readCSV = csv.reader(csvfile, delimiter=',')
			y = np.zeros((1,30))
			for idx, row in enumerate(readCSV):
				array = np.array(row)
				try:
					i =0
					while i<num_video[idx]:
						y = np.vstack((y, np.expand_dims(array[1:], axis=0)))
						i+=1
				except:
					full_array = np.zeros((1,30))
					for i in range(len(array[1:])):
						try:
							full_array[i] = array[1:][i]
						except:
							pass
					i = 0
					while i<num_video[idx]:
						y = np.vstack((y, np.expand_dims(full_array, axis=0)))
						i+=1
			y = y[1:]
			print 'y shape: ', y.shape
			return y
	elif file_name == 'ground_truth_v.csv':
		with open(join(label_dir,file_name)) as csvfile:
			readCSV = csv.reader(csvfile, delimiter=',')
			y = np.zeros((1,966))
			for idx, row in enumerate(readCSV):
				array = np.array(row)
				i =0
				while i<num_video[idx]:
					y = np.vstack((y, np.expand_dims(array[1:], axis=0)))
					i+=1
			y = y[1:]
			print 'y shape: ', y.shape
			return y
	else:
		print 'The file name for label is illegal'


# def load_x(data_dir,num_video):
num_video = video_number(data_dir)
"""load the feature data.
Args:
	data_dir: The path to all the folders.
Returns:
	A numpy array file of all the video features. The dimention should be sum(num_video) x num_classes"""
fold_dir = data_dir
fold_names = [f for f in os.listdir(fold_dir)]
sort_name = np.sort(fold_names)
x = np.zeros((1,2048))
for id, dir in enumerate(sort_name[1:]):
	file_dir = join(fold_dir,dir)
	for name in os.listdir(file_dir):
		feature = np.load(join(file_dir,name)) 
		print feature.shape
		i = 0
		while i<num_video[id]:
			y = np.vstack((x, np.expand_dims(feature, axis=0)))
			i+=1
		print x.shape



num_video = video_number(data_dir)
y = load_y(label_dir, 'ground_truth.csv', num_video)
print y.shape







